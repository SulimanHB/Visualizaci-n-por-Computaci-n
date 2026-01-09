import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
import cv2
import numpy as np
from PIL import Image
import logging

# Suppress HuggingFace warnings for cleaner output
logging.getLogger("diffusers").setLevel(logging.ERROR)

# Device Configuration
device = "cuda"
dtype = torch.float16

print(f"--- LOADING TURBO ENGINE ON: {torch.cuda.get_device_name(0)} ---")

# 1. Load Scribble ControlNet (Ideal for sketches)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble",
    torch_dtype=dtype
).to(device)

# 2. Base Pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False
).to(device)

# 3. Enable LCM (Turbo Mode)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

def improve_draw_ia(canvas_np):
    # --- PRE-PROCESSING ---
    # Invert colors for Scribble (Black lines on white background)
    img_gray = cv2.cvtColor(canvas_np, cv2.COLOR_BGR2GRAY)
    img_inverted = cv2.bitwise_not(img_gray)
    control_image = Image.fromarray(img_inverted).resize((512, 512))

    # --- STRATEGIC PROMPTS ---
    
    # Positive Prompt
    prompt = (
        "isolated graffiti element on black background, "
        "urban street art style, "
        "paint drips, aerosol texture, highly detailed, "
        "floating object, no background"
    )

    # Negative Prompt
    negative_prompt = (
        "wall, brick wall, concrete, background texture, "
        "square frame, picture frame, cropped, low quality, "
        "blurry, ugly, paper, white background"
    )
    
    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=6,       # Turbo Speed
            guidance_scale=1.5,          # Low guidance for LCM
            controlnet_conditioning_scale=1.0, 
            cross_attention_kwargs={"scale": 1.0}
        ).images[0]

    # Post-processing
    res = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
    
    # Cleanup: Force dark grey pixels to pure black for transparency
    res[np.all(res < [30, 30, 30], axis=-1)] = [0, 0, 0]

    return cv2.resize(res, (1000, 1000))