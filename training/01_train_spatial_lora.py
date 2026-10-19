"""
training/01_train_spatial_lora.py

Phase 1 Training: Spatial LoRA Adaptation on Stable Diffusion 1.5.
Trains LoRA adapters on SD 1.5 UNet spatial cross-attention (to_k, to_v, to_q)
and self-attention projections using static frames extracted from historical
atmospheric nuclear test clips.

This grounds visual conditioning:
- Fireball geometry and nuclear cloud textures
- Pacific Proving Grounds ocean lagoons vs. Nevada Test Site desert flats
- 16mm archival film grain, scratches, and monochrome/Technicolor film look

Usage:
  python training/01_train_spatial_lora.py --config training/config.yaml --max_train_steps 10 --device cpu
  python training/01_train_spatial_lora.py --config training/config.yaml
"""

import os
import sys
import argparse
import yaml
import math
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.dataset import NuclearTestDataset


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_spatial_lora(unet, rank=16, alpha=16, target_modules=None):
    """
    Injects LoRA into UNet attention projections.
    Tries peft first, or diffusers native LoRA processor as fallback.
    """
    if target_modules is None:
        target_modules = ["to_k", "to_v", "to_q", "to_out.0"]
        
    try:
        from peft import LoraConfig, get_peft_model
        print(f"Injecting PEFT LoRA (rank={rank}, alpha={alpha}) into UNet targets: {target_modules}")
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
        print("Note: 'peft' not installed. Adding LoRA via Diffusers AttnProcessor...")
        from diffusers.loaders import AttnProcsLayers
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
        lora_layers = AttnProcsLayers(unet.attn_processors)
        return lora_layers, "diffusers"


def train_spatial_lora(args):
    config = load_config(args.config)
    output_dir = os.path.join(PROJECT_ROOT, config["project"]["output_dir"], "phase1_spatial")
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Running Phase 1 (Spatial LoRA) on device: {device}")
    
    model_id = config["models"]["pretrained_model_name_or_path"]
    
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
    
    # 3. Load UNet
    print(f"Loading SD 1.5 UNet from {model_id}...")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    unet.requires_grad_(False)
    
    p1_cfg = config["phase1_spatial"]
    trainable_model, lora_type = setup_spatial_lora(
        unet,
        rank=p1_cfg.get("lora_rank", 16),
        alpha=p1_cfg.get("lora_alpha", 16),
        target_modules=p1_cfg.get("target_modules", ["to_k", "to_v", "to_q", "to_out.0"])
    )
    
    if lora_type == "peft":
        unet.to(device)
        trainable_params = filter(lambda p: p.requires_grad, unet.parameters())
    else:
        unet.to(device)
        trainable_model.to(device)
        trainable_params = trainable_model.parameters()
        
    # 4. Dataset & DataLoader
    print("Initializing NuclearTestDataset (mode='spatial')...")
    train_dataset = NuclearTestDataset(
        manifest_path=config["data"]["manifest_path"],
        split="train",
        tokenizer=tokenizer,
        mode="spatial",
        resolution=config["data"]["resolution"],
        load_latents=True
    )
    
    batch_size = p1_cfg.get("train_batch_size", 2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 5. Optimizer & LR Scheduler
    learning_rate = float(p1_cfg.get("learning_rate", 1e-4))
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-2)
    
    max_steps = args.max_train_steps if args.max_train_steps else p1_cfg.get("max_train_steps", 2000)
    lr_scheduler = get_scheduler(
        p1_cfg.get("lr_scheduler", "constant_with_warmup"),
        optimizer=optimizer,
        num_warmup_steps=p1_cfg.get("lr_warmup_steps", 100),
        num_training_steps=max_steps
    )
    
    # 6. Training Loop
    print("\n" + "=" * 65)
    print("      STARTING PHASE 1: SPATIAL LORA TRAINING")
    print("=" * 65)
    print(f"Total training steps: {max_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    global_step = 0
    progress_bar = tqdm(total=max_steps, desc="Phase 1 Steps")
    
    while global_step < max_steps:
        for batch in train_loader:
            if global_step >= max_steps:
                break
                
            # Get conditioning latents and text embeddings
            data = batch["data"].to(device=device, dtype=torch.float32)
            input_ids = batch["input_ids"].to(device)
            
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]
                
            # Sample noise
            noise = torch.randn_like(data)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (data.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(data, noise, timesteps)
            
            # Predict noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # Compute MSE loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})
            
            # Periodic Checkpoint
            save_steps = p1_cfg.get("save_steps", 250)
            if global_step % save_steps == 0 or global_step == max_steps:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                if lora_type == "peft":
                    unet.save_pretrained(ckpt_dir)
                else:
                    unet.save_attn_procs(ckpt_dir)
                print(f"\n  [Checkpoint] Saved Spatial LoRA to {ckpt_dir}")
                
    progress_bar.close()
    
    # Save final Spatial LoRA
    final_dir = os.path.join(output_dir, "final_spatial_lora")
    os.makedirs(final_dir, exist_ok=True)
    if lora_type == "peft":
        unet.save_pretrained(final_dir)
    else:
        unet.save_attn_procs(final_dir)
        
    print(f"\nPhase 1 Complete! Final Spatial LoRA saved to: {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Spatial LoRA Training Engine")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to config.yaml")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()
    
    train_spatial_lora(args)


if __name__ == "__main__":
    main()
