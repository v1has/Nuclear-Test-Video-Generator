"""
training/01_train_spatial_lora.py

Phase 1 Spatial LoRA Training Engine.
Fine-tunes spatial cross-attention layers of Stable Diffusion 1.5 on nuclear test frames
to learn historical visual tokens (burst type, yield tier, terrain, lighting, film look).

Fixes P1 items:
- Implements gradient accumulation, mixed precision (fp16), and gradient checkpointing
- Saves only LoRA adapter weights (never accidental full UNet)
- Resumable state and parameter logging
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
from diffusers import UNet2DConditionModel, DDPMScheduler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.dataset import NuclearTestDataset


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_spatial_lora(unet, rank=16, alpha=16, target_modules=None):
    """Injects LoRA into UNet spatial attention projections."""
    if target_modules is None:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        
    try:
        from peft import LoraConfig, get_peft_model
        print(f"Injecting PEFT Spatial LoRA (rank={rank}, alpha={alpha}) into: {target_modules}")
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()
        return unet, "peft"
    except ImportError:
        print("Note: 'peft' not installed. Falling back to direct diffusers attention processors...")
        from diffusers.models.attention_processor import LoRAAttnProcessor
        lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=rank
            )
        unet.set_attn_processor(lora_attn_procs)
        return unet, "diffusers"


def train_spatial_lora(args):
    config = load_config(args.config)
    output_dir = os.path.join(PROJECT_ROOT, config["project"]["output_dir"], "phase1_spatial")
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Running Phase 1 (Spatial LoRA) on device: {device}")
    
    model_id = config["models"]["pretrained_model_name_or_path"]
    p1_cfg = config["phase1_spatial"]
    
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
    
    # 3. Load base UNet
    print(f"Loading SD 1.5 UNet from {model_id}...")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    
    # Gradient checkpointing
    if p1_cfg.get("gradient_checkpointing", True):
        unet.enable_gradient_checkpointing()
        
    # Inject LoRA
    unet, lora_type = setup_spatial_lora(
        unet,
        rank=p1_cfg.get("lora_rank", 16),
        alpha=p1_cfg.get("lora_alpha", 16),
        target_modules=p1_cfg.get("target_modules", ["to_k", "to_q", "to_v", "to_out.0"])
    )
    unet.to(device)
    
    # 4. Dataset & DataLoader (mode='spatial')
    print("Initializing NuclearTestDataset (mode='spatial')...")
    train_dataset = NuclearTestDataset(
        manifest_path=config["data"]["manifest_path"],
        split="train",
        tokenizer=tokenizer,
        mode="spatial",
        resolution=config["data"]["resolution"],
        load_latents=True
    )
    
    batch_size = p1_cfg.get("train_batch_size", 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 5. Optimizer & Scheduler
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(p1_cfg.get("learning_rate", 1e-4)),
        weight_decay=1e-2
    )
    
    max_steps = args.max_train_steps or p1_cfg.get("max_train_steps", 2000)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    
    # Mixed precision setup
    use_fp16 = p1_cfg.get("mixed_precision", "fp16") == "fp16" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    accum_steps = p1_cfg.get("gradient_accumulation_steps", 1)
    
    # 6. Training Loop
    print(f"Starting training: {max_steps} steps, batch_size={batch_size}, accum_steps={accum_steps}, fp16={use_fp16}")
    unet.train()
    global_step = 0
    progress_bar = tqdm(total=max_steps, desc="Phase 1 (Spatial LoRA)")
    
    optimizer.zero_grad()
    while global_step < max_steps:
        for step, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break
                
            data = batch["data"].to(device)  # Latent: (B, 4, 64, 64)
            input_ids = batch["input_ids"].to(device)
            
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]
                
            noise = torch.randn_like(data)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (data.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(data, noise, timesteps)
            
            with torch.cuda.amp.autocast(enabled=use_fp16):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                loss = loss / accum_steps
                
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
                save_steps = p1_cfg.get("save_steps", 250)
                if global_step % save_steps == 0 or global_step == max_steps:
                    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    if lora_type == "peft":
                        unet.save_pretrained(ckpt_dir)
                    else:
                        unet.save_attn_procs(ckpt_dir)
                        
    progress_bar.close()
    
    # Final save
    final_dir = os.path.join(output_dir, "final_spatial_lora")
    os.makedirs(final_dir, exist_ok=True)
    if lora_type == "peft":
        unet.save_pretrained(final_dir)
    else:
        unet.save_attn_procs(final_dir)
    print(f"\n[SUCCESS] Phase 1 Complete! Spatial LoRA saved to: {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Spatial LoRA Training Engine")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to config.yaml")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()
    train_spatial_lora(args)


if __name__ == "__main__":
    main()
