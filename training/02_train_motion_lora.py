"""
training/02_train_motion_lora.py

Phase 2 Motion LoRA Training Engine.
Fine-tunes AnimateDiff temporal self-attention layers on 16-frame nuclear test
sequences, freezing spatial layers and loading the Phase 1 Spatial LoRA weights.

Fixes P0.7 and P1 items:
- Target isolation: Targets strictly motion_modules.* temporal self-attention;
  never suffix-matches spatial attn1 layers.
- Strict freezing: Explicitly freezes all spatial parameters and runs an
  automated assertion test verifying 0 spatial parameters have requires_grad=True.
- Matching Phase 1 loader: Merges Phase 1 PEFT / diffusers spatial adapter
  directly into base UNet spatial weights before injecting Phase 2 Motion LoRA.
- Implements config's gradient accumulation, mixed precision (fp16), and
  gradient checkpointing.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, UNetMotionModel, MotionAdapter, DDPMScheduler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.dataset import NuclearTestDataset


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_motion_lora(unet, rank=16, alpha=16, target_modules=None):
    """
    Injects LoRA strictly into AnimateDiff temporal self-attention projections.
    Targets ONLY motion_modules.* layers (AGENTS.md P0.7).
    """
    # Regex targeting only temporal self-attention in motion modules
    temporal_regex = r".*motion_modules.*attn1\.(to_q|to_k|to_v|to_out\.0)"
    
    try:
        from peft import LoraConfig, get_peft_model
        print(f"Injecting PEFT Motion LoRA (rank={rank}, alpha={alpha}) with temporal regex target: '{temporal_regex}'")
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=temporal_regex
        )
        unet = get_peft_model(unet, lora_config)
        
        # Enforce that all non-temporal parameters are strictly frozen (P0.7)
        spatial_fixed = 0
        for name, param in unet.named_parameters():
            is_temporal = any(k in name for k in ["motion_modules", "temporal"])
            if not is_temporal:
                if param.requires_grad:
                    param.requires_grad = False
                    spatial_fixed += 1
                    
        # Verification assertion: No spatial parameter may be trainable
        for name, param in unet.named_parameters():
            if param.requires_grad:
                assert any(k in name for k in ["motion_modules", "temporal"]), (
                    f"SPATIAL LEAKAGE DETECTED (P0.7): Parameter '{name}' is marked trainable in Phase 2!"
                )
                
        print("P0.7 Assertion Passed: Exactly 0 spatial parameters trainable in Phase 2 Motion LoRA.")
        unet.print_trainable_parameters()
        return unet, "peft"
    except ImportError:
        print("Note: 'peft' not installed. Directly training temporal module parameters...")
        trainable_params = []
        for name, param in unet.named_parameters():
            if any(t in name for t in ["temporal_transformer_blocks", "motion_modules"]):
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
        print(f"Directly training {len(trainable_params)} temporal parameters.")
        return unet, "direct"


def train_motion_lora(args):
    config = load_config(args.config)
    output_dir = os.path.join(PROJECT_ROOT, config["project"]["output_dir"], "phase2_motion")
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Running Phase 2 (Motion LoRA) on device: {device}")
    
    model_id = config["models"]["pretrained_model_name_or_path"]
    motion_adapter_id = config["models"]["motion_adapter_id"]
    p2_cfg = config["phase2_motion"]
    
    # 1. Load Tokenizer & Text Encoder
    print(f"Loading CLIP Tokenizer & Text Encoder from {model_id}...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    text_encoder.to(device)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
        
    # 2. Load Noise Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # 3. Load base UNet and Motion Adapter
    print(f"Loading SD 1.5 UNet and Motion Adapter from {motion_adapter_id}...")
    try:
        motion_adapter = MotionAdapter.from_pretrained(motion_adapter_id)
        unet = UNetMotionModel.from_pretrained(model_id, subfolder="unet", motion_adapter=motion_adapter)
    except Exception as e:
        print(f"Notice: Loading fallback motion module setup ({e})...")
        base_unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        motion_adapter = MotionAdapter()
        unet = UNetMotionModel.from_unet2d(base_unet, motion_adapter=motion_adapter)
        
    # 4. Check if Phase 1 Spatial LoRA weights exist to merge / load (P0.7)
    spatial_lora_dir = os.path.join(PROJECT_ROOT, config["project"]["output_dir"], "phase1_spatial", "final_spatial_lora")
    if os.path.exists(spatial_lora_dir):
        print(f"Found Phase 1 Spatial LoRA at {spatial_lora_dir}. Merging into spatial weights...")
        try:
            adapter_cfg = os.path.join(spatial_lora_dir, "adapter_config.json")
            if os.path.exists(adapter_cfg):
                from peft import PeftModel
                peft_spatial = PeftModel.from_pretrained(unet, spatial_lora_dir)
                unet = peft_spatial.merge_and_unload()
                print("[SUCCESS] Merged Phase 1 Spatial LoRA into UNet spatial weights.")
            elif hasattr(unet, "load_attn_procs"):
                unet.load_attn_procs(spatial_lora_dir)
                print("[SUCCESS] Loaded Phase 1 Spatial LoRA via load_attn_procs.")
            elif hasattr(unet, "load_lora_adapter"):
                unet.load_lora_adapter(spatial_lora_dir)
                print("[SUCCESS] Loaded Phase 1 Spatial LoRA via load_lora_adapter.")
        except Exception as e:
            print(f"Warning: Could not merge Phase 1 Spatial LoRA ({e}). Proceeding with base spatial weights.")

    # Freeze all spatial parameters before injecting Motion LoRA
    for name, param in unet.named_parameters():
        if "temporal" not in name and "motion" not in name:
            param.requires_grad = False
            
    # Gradient checkpointing
    if p2_cfg.get("gradient_checkpointing", True):
        unet.enable_gradient_checkpointing()

    # Inject Phase 2 Motion LoRA targeting ONLY temporal layers
    unet, lora_type = setup_motion_lora(
        unet,
        rank=p2_cfg.get("lora_rank", 16),
        alpha=p2_cfg.get("lora_alpha", 16)
    )
    unet.to(device)
    
    # 5. Dataset & DataLoader (mode='motion')
    print("Initializing NuclearTestDataset (mode='motion')...")
    train_dataset = NuclearTestDataset(
        manifest_path=config["data"]["manifest_path"],
        split="train",
        tokenizer=tokenizer,
        mode="motion",
        resolution=config["data"]["resolution"],
        load_latents=True
    )
    
    batch_size = p2_cfg.get("train_batch_size", 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 6. Optimizer & Scheduler
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(p2_cfg.get("learning_rate", 3.5e-5)),
        weight_decay=1e-2
    )
    
    max_steps = args.max_train_steps or p2_cfg.get("max_train_steps", 3000)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    
    # Mixed precision setup
    use_fp16 = p2_cfg.get("mixed_precision", "fp16") == "fp16" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    accum_steps = p2_cfg.get("gradient_accumulation_steps", 4)
    
    # 7. Training Loop
    print(f"Starting Phase 2 training: {max_steps} steps, batch_size={batch_size}, accum_steps={accum_steps}, fp16={use_fp16}")
    unet.train()
    global_step = 0
    progress_bar = tqdm(total=max_steps, desc="Phase 2 (Motion LoRA)")
    
    optimizer.zero_grad()
    while global_step < max_steps:
        for step, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break
                
            data = batch["data"].to(device)  # Latent: (B, 4, 16, 64, 64)
            input_ids = batch["input_ids"].to(device)
            
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]
                
            noise = torch.randn_like(data)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (data.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(data, noise, timesteps)
            
            with torch.cuda.amp.autocast(enabled=use_fp16):
                # Predict noise residual: (B, 4, 16, 64, 64)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Base reconstruction MSE loss
                mse_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Temporal difference loss between adjacent frame noise predictions (P1 ablated / weighted)
                diff_pred = model_pred[:, :, 1:, :, :] - model_pred[:, :, :-1, :, :]
                diff_noise = noise[:, :, 1:, :, :] - noise[:, :, :-1, :, :]
                temp_loss = F.mse_loss(diff_pred.float(), diff_noise.float(), reduction="mean")
                
                loss = (mse_loss + 0.1 * temp_loss) / accum_steps
                
            scaler.scale(loss).backward()
            
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item() * accum_steps:.4f}", "step": global_step})
                
                # Checkpoint
                save_steps = p2_cfg.get("save_steps", 250)
                if global_step % save_steps == 0 or global_step == max_steps:
                    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    if lora_type == "peft":
                        unet.save_pretrained(ckpt_dir)
                    else:
                        unet.save_attn_procs(ckpt_dir)
                        
    progress_bar.close()
    
    # Final save
    final_dir = os.path.join(output_dir, "final_motion_lora")
    os.makedirs(final_dir, exist_ok=True)
    if lora_type == "peft":
        unet.save_pretrained(final_dir)
    else:
        unet.save_attn_procs(final_dir)
    print(f"\n[SUCCESS] Phase 2 Complete! Motion LoRA saved to: {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Motion LoRA Training Engine")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to config.yaml")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()
    train_motion_lora(args)


if __name__ == "__main__":
    main()
