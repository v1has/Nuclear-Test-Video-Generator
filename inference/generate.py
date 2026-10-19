"""
inference/generate.py

Inference generation engine for parameter-controlled archival nuclear test videos.
Synthesizes 16-frame, 512x512 historical video clips conditioned on:
- burst_type (air-burst, surface-tower, surface-barge, underwater)
- yield_tier (sub-kiloton, tactical-yield, intermediate-yield, megaton-class)
- terrain (desert-flats, tropical-ocean, low-mountains)
- lighting (day, night, dawn)
- phase (early-fireball, expansion, stabilized-cloud)
- film_look (black-and-white, faded-technicolor, colour-restored)
- smoke_trails (bool)

Fixes P0.10 and P0.5 blockers:
- Validates prompt tokens against canonical schema.py
- Eliminates silent simulation: requires explicit '--mode simulation'
- Simulation frames watermarked with visible '[SIMULATION - SYNTHETIC PLACEHOLDER]'
- Explicit multi-adapter activation via pipe.set_adapters(['spatial', 'motion'])
- Configurable width and height
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import schema


def construct_prompt(burst_type, yield_tier, terrain, lighting, phase, film_look, smoke_trails):
    """Construct and validate conditioning prompt using the canonical schema."""
    prompt = schema.format_prompt(
        burst_type=burst_type,
        yield_tier=yield_tier,
        terrain=terrain,
        lighting=lighting,
        phase=phase,
        film_look=film_look,
        smoke_trails=smoke_trails
    )
    schema.validate_prompt(prompt)
    
    negative_prompt = (
        "modern digital rendering, 3d cgi, lowres, text, watermark, signature, "
        "oversaturated, cartoon, anime, blurry, flickering, jittery, distorted"
    )
    return prompt, negative_prompt


def run_physics_simulation(args, prompt, out_path):
    """
    Renders deterministic physics simulation preview for verification without GPU.
    Clearly marks and watermarks every frame as a synthetic simulation.
    """
    print("[SIMULATION] Generating physics-based preview animation...")
    np.random.seed(args.seed)
    
    width = args.width
    height = args.height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, args.fps, (width, height))
    
    is_bw = "black-and-white" in args.film_look
    is_underwater = args.burst_type == "underwater"
    is_desert = "desert" in args.terrain
    
    sky_color = [180, 180, 180] if is_bw else ([210, 160, 100] if is_desert else [220, 170, 110])
    ground_color = [100, 100, 100] if is_bw else ([80, 140, 180] if is_desert else [140, 90, 40])
    
    for t in range(args.num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        horizon = int(height * 0.7)
        frame[:horizon, :] = sky_color
        frame[horizon:, :] = ground_color
        
        progress = (t + 1) / args.num_frames
        center_x = width // 2
        center_y = int(height * 0.62 - progress * (height * 0.18))
        radius = int(width * 0.05 + progress * (width * 0.22))
        
        if is_underwater:
            col_w = int(width * 0.06 + progress * (width * 0.08))
            col_top = max(int(height * 0.2), int(horizon - progress * (height * 0.4)))
            cv2.rectangle(frame, (center_x - col_w, col_top), (center_x + col_w, horizon), (240, 240, 240) if is_bw else (230, 240, 230), -1)
            cv2.circle(frame, (center_x, col_top), int(radius * 0.8), (250, 250, 250), -1)
            surge_w = int(progress * (width * 0.44))
            cv2.ellipse(frame, (center_x, horizon), (surge_w, int(surge_w * 0.25)), 0, 0, 360, (230, 230, 230), -1)
        else:
            cloud_color = (245, 245, 245) if (is_bw or args.phase == "stabilized-cloud") else (40, 150, 255)
            stem_w = int(width * 0.03 + progress * (width * 0.05))
            cv2.rectangle(frame, (center_x - stem_w, center_y), (center_x + stem_w, horizon), (180, 180, 180) if is_bw else (120, 120, 120), -1)
            cv2.ellipse(frame, (center_x, center_y), (radius, int(radius * 0.65)), 0, 0, 360, cloud_color, -1)
            
        if args.smoke_trails:
            for off in [-160, -110, -60, 60, 110, 160]:
                scaled_off = int(off * (width / 512))
                cv2.line(frame, (center_x + scaled_off, 40), (center_x + scaled_off, horizon), (240, 240, 240), 2)
                
        grain = np.random.normal(0, 15, (height, width, 3)).astype(np.int16)
        noisy = np.clip(frame.astype(np.int16) + grain, 0, 255).astype(np.uint8)
        
        # Prominent simulation watermark (AGENTS.md P0.10)
        cv2.putText(noisy, "[SIMULATION - SYNTHETIC PLACEHOLDER]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(noisy, f"{args.burst_type.upper()} | {args.yield_tier} | {args.phase} (F:{t+1:02d})", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        vw.write(noisy)
        
    vw.release()
    print(f"\n[SIMULATION SUCCESS] Saved synthetic simulation preview video to: {out_path}")
    print("Notice: Output clearly watermarked as simulation placeholder.")
    return out_path


def generate_video(args):
    # Resolve terrain and lighting (supporting legacy --setting)
    if hasattr(args, "setting") and args.setting and not args.terrain:
        terrain, lighting = schema.normalize_terrain_and_lighting(args.setting, args.lighting)
    else:
        terrain = args.terrain or "desert-flats"
        lighting = args.lighting or "day"
        
    prompt, negative_prompt = construct_prompt(
        args.burst_type,
        args.yield_tier,
        terrain,
        lighting,
        args.phase,
        args.film_look,
        args.smoke_trails
    )
    
    print("=" * 65)
    print("      NUCLEAR TEST VIDEO GENERATOR - INFERENCE")
    print("=" * 65)
    print(f"Conditioning Prompt:\n  \"{prompt}\"")
    print(f"Negative Prompt:\n  \"{negative_prompt}\"")
    print(f"Resolution: {args.width}x{args.height} | Frames: {args.num_frames} @ {args.fps} fps")
    print(f"Steps: {args.steps} | Guidance: {args.guidance_scale} | Seed: {args.seed}")
    print(f"Mode: {args.mode}")
    
    out_path = os.path.join(PROJECT_ROOT, args.output) if not os.path.isabs(args.output) else args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Explicit simulation mode check
    if args.mode == "simulation":
        return run_physics_simulation(args, prompt, out_path)
        
    # Diffusion mode
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}")
    
    if not torch.cuda.is_available() and device == "cpu" and not args.allow_cpu_diffusion:
        raise RuntimeError(
            "CUDA GPU is not available for real diffusion inference.\n"
            "Silent fallback to simulation is strictly forbidden (AGENTS.md P0.10).\n"
            "Options:\n"
            "  1. Run on a GPU machine (Kaggle / Colab / cloud)\n"
            "  2. Explicitly test the preview simulation with: --mode simulation\n"
            "  3. Force CPU diffusion (very slow) with: --allow_cpu_diffusion"
        )
        
    try:
        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    except ImportError as e:
        raise ImportError(f"diffusers / torch not fully installed for diffusion inference: {e}")
        
    print("Initializing AnimateDiff pipeline...")
    adapter_id = "guoyww/animatediff-motion-adapter"
    adapter = MotionAdapter.from_pretrained(
        adapter_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        motion_adapter=adapter,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
    
    loaded_adapters = []
    adapter_weights = []
    
    # Load Spatial LoRA
    if args.spatial_lora and os.path.exists(args.spatial_lora):
        print(f"Loading Phase 1 Spatial LoRA from {args.spatial_lora}...")
        pipe.load_lora_weights(args.spatial_lora, adapter_name="spatial")
        loaded_adapters.append("spatial")
        adapter_weights.append(float(args.spatial_scale))
        
    # Load Motion LoRA
    if args.motion_lora and os.path.exists(args.motion_lora):
        print(f"Loading Phase 2 Motion LoRA from {args.motion_lora}...")
        pipe.load_lora_weights(args.motion_lora, adapter_name="motion")
        loaded_adapters.append("motion")
        adapter_weights.append(float(args.motion_scale))
        
    # Explicitly activate multiple adapters together (AGENTS.md P0.10)
    if loaded_adapters:
        print(f"Activating LoRA adapters: {loaded_adapters} with weights {adapter_weights}")
        pipe.set_adapters(loaded_adapters, adapter_weights=adapter_weights)
        
    pipe.to(device)
    
    generator = torch.Generator(device=device).manual_seed(args.seed)
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        generator=generator
    )
    frames = output.frames[0]
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, args.fps, (args.width, args.height))
    for img in frames:
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        vw.write(cv2.resize(bgr, (args.width, args.height)))
    vw.release()
    
    print(f"\n[SUCCESS] Generated diffusion video saved to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Nuclear Test Archival Video Generator - Inference")
    parser.add_argument("--burst_type", type=str, default="surface-tower", choices=schema.BURST_TYPES, help="Burst detonation type")
    parser.add_argument("--yield_tier", type=str, default="tactical-yield", choices=schema.YIELD_TIERS, help="Yield energy tier")
    parser.add_argument("--terrain", type=str, default="desert-flats", choices=schema.TERRAINS, help="Terrain environment")
    parser.add_argument("--lighting", type=str, default="day", choices=schema.LIGHTINGS, help="Lighting condition")
    parser.add_argument("--setting", type=str, default=None, help="Legacy setting string (e.g. desert-day, tropical-ocean)")
    parser.add_argument("--phase", type=str, default="expansion", choices=schema.PHASES, help="Detonation evolution phase")
    parser.add_argument("--film_look", type=str, default="black-and-white", choices=schema.FILM_LOOKS, help="Film archival aesthetic")
    parser.add_argument("--smoke_trails", action="store_true", help="Include diagnostic smoke rocket trails")
    
    parser.add_argument("--output", type=str, default="samples/generated_sample.mp4", help="Output MP4 file path")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames (default 16)")
    parser.add_argument("--width", type=int, default=512, help="Frame width (default 512)")
    parser.add_argument("--height", type=int, default=512, help="Frame height (default 512)")
    parser.add_argument("--fps", type=int, default=8, help="Output frame rate (default 8)")
    parser.add_argument("--steps", type=int, default=25, help="Number of diffusion inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    
    parser.add_argument("--spatial_lora", type=str, default=None, help="Path to Phase 1 Spatial LoRA weights")
    parser.add_argument("--motion_lora", type=str, default=None, help="Path to Phase 2 Motion LoRA weights")
    parser.add_argument("--spatial_scale", type=float, default=1.0, help="Spatial LoRA adapter weight scale")
    parser.add_argument("--motion_scale", type=float, default=1.0, help="Motion LoRA adapter weight scale")
    
    # Modes
    parser.add_argument("--mode", type=str, default="diffusion", choices=["diffusion", "simulation"], help="Mode ('diffusion' or 'simulation')")
    parser.add_argument("--allow_cpu_diffusion", action="store_true", help="Allow running diffusion on CPU")
    
    args = parser.parse_args()
    generate_video(args)


if __name__ == "__main__":
    main()
