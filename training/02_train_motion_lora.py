"""
training/02_train_motion_lora.py

Phase 2 Training: Motion LoRA Adaptation on AnimateDiff Motion Module.
Trains LoRA adapters targeting temporal self-attention layers in the AnimateDiff
motion module, while freezing the spatial UNet (with the Phase 1 Spatial LoRA active/merged).

This grounds fluid physical dynamics:
- Rapid fireball shockwave expansion (~Mach 3 to subsonic transition)
- Toroidal vortex formation and stem suction
- Condensation (Wilson) cloud flash and clearing
- Wave surging and base surge progression in shallow-water bursts

Input:
- Pre-encoded 16-frame latent tensors: (4, 16, H/8, W/8)
- Tokenized text conditioning prompts

Usage:
  python training/02_train_motion_lora.py --config training/config.yaml --max_train_steps 10 --device cpu
  python training/02_train_motion_lora.py --config training/config.yaml
"""

import os
import sys
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    UNet2DConditionModel,
    UNetMotionModel,
    MotionAdapter,
    DDPMScheduler
)
from diffusers.optimization import get_scheduler

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.dataset import NuclearTestDataset


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_motion_lora(unet, rank=16, alpha=16, target_modules=None):
    """
    Injects LoRA into AnimateDiff temporal self-attention projections.
    Targeting temporal self-attention layers (attn1.to_q, attn1.to_k, attn1.to_v, attn1.to_out.0).
    """
    if target_modules is None:
        target_modules = ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"]
        
    try:
        from peft import LoraConfig, get_peft_model
        print(f"Injecting PEFT Motion LoRA (rank={rank}, alpha={alpha}) into temporal attention targets: {target_modules}")
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
        print("Note: 'peft' not installed. Setting up trainable parameters directly on temporal modules...")
        # Train the temporal transformer block parameters directly
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
        
    # 4. Check if Phase 1 Spatial LoRA weights exist to merge / load
    spatial_lora_dir = os.path.join(PROJECT_ROOT, config["project"]["output_dir"], "phase1_spatial", "final_spatial_lora")
    if os.path.exists(spatial_lora_dir):
        print(f"Found Phase 1 Spatial LoRA at {spatial_lora_dir}. Merging into spatial weights...")
        try:
            if hasattr(unet, "load_attn_procs"):
                unet.load_attn_procs(spatial_lora_dir)
        except Exception as e:
            print(f"Note on Spatial LoRA loading: {e}")
            
    # Freeze all spatial parameters
    for name, param in unet.named_parameters():
        if "temporal" not in name and "motion" not in name:
            param.requires_grad = False
            
    p2_cfg = config["phase2_motion"]
    unet, lora_type = setup_motion_lora(
        unet,
        rank=p2_cfg.get("lora_rank", 16),
        alpha=p2_cfg.get("lora_alpha", 16),
        target_modules=p2_cfg.get("target_modules", ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"])
    )
    unet.to(device)
    
    # 5. Dataset & DataLoader (mode='motion', loads 16-frame sequences)
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
    trainable_params = filter(lambda p: p.requires_grad, unet.parameters())
    learning_rate = float(p2_cfg.get("learning_rate", 3.5e-5))
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-2)
    
    max_steps = args.max_train_steps if args.max_train_steps else p2_cfg.get("max_train_steps", 3000)
    lr_scheduler = get_scheduler(
        p2_cfg.get("lr_scheduler", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=p2_cfg.get("lr_warmup_steps", 150),
        num_training_steps=max_steps
    )
    
    # 7. Training Loop
    print("\n" + "=" * 65)
    print("      STARTING PHASE 2: MOTION LORA TRAINING")
    print("=" * 65)
    print(f"Total training steps: {max_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    global_step = 0
    progress_bar = tqdm(total=max_steps, desc="Phase 2 Steps")
    
    while global_step < max_steps:
        for batch in train_loader:
            if global_step >= max_steps:
                break
                
            # Latent sequence data: (B, 4, 16, 64, 64)
            data = batch["data"].to(device=device, dtype=torch.float32)
            input_ids = batch["input_ids"].to(device)
            
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]
                
            # Sample noise of same shape (B, 4, 16, 64, 64)
            noise = torch.randn_like(data)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (data.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(data, noise, timesteps)
            
            # Predict noise residual with UNetMotionModel
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # Temporal motion loss: combination of MSE on noise + inter-frame delta loss
            loss_noise = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            # Temporal difference loss (encourages smooth inter-frame kinematics)
            if model_pred.shape[2] > 1:
                pred_delta = model_pred[:, :, 1:, :, :] - model_pred[:, :, :-1, :, :]
                noise_delta = noise[:, :, 1:, :, :] - noise[:, :, :-1, :, :]
                loss_delta = F.mse_loss(pred_delta.float(), noise_delta.float(), reduction="mean")
                loss = loss_noise + 0.1 * loss_delta
            else:
                loss = loss_noise
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})
            
            # Periodic Checkpoint
            save_steps = p2_cfg.get("save_steps", 250)
            if global_step % save_steps == 0 or global_step == max_steps:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                if lora_type == "peft":
                    unet.save_pretrained(ckpt_dir)
                else:
                    torch.save(unet.motion_modules.state_dict(), os.path.join(ckpt_dir, "motion_module.pt"))
                print(f"\n  [Checkpoint] Saved Motion LoRA to {ckpt_dir}")
                
    progress_bar.close()
    
    # Save final Motion LoRA
    final_dir = os.path.join(output_dir, "final_motion_lora")
    os.makedirs(final_dir, exist_ok=True)
    if lora_type == "peft":
        unet.save_pretrained(final_dir)
    else:
        torch.save(unet.motion_modules.state_dict(), os.path.join(final_dir, "motion_module.pt"))
        
    print(f"\nPhase 2 Complete! Final Motion LoRA saved to: {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Motion LoRA Training Engine")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to config.yaml")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()
    
    train_motion_lora(args)


if __name__ == "__main__":
    main()
