import argparse
import os
import time

import torch
from data import encode_text, load_checkpoint
from inference import inference
from transfusion import Transfusion
from vae import vae_encode
from wrapper import ModelWrapper
from transformers import AutoTokenizer
from diffusers import AutoencoderKL
from accelerate import Accelerator
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the Transfusion model')
    parser.add_argument('--checkpoint_name', type=str, default='checkpoints/model_checkpoint_epoch_9_step_77000.pth', help='Load checkpoint')
    parser.add_argument('--model_name', type=str, default='HuggingFaceTB/SmolLM-1.7B', help='Name of the model to use')
    parser.add_argument('--image_size', type=int, default=256, help='Image size for training')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size for training')
    parser.add_argument('--max_length', type=int, default=128, help='Max length for text')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps for inference')
    parser.add_argument('--timestep', type=float, default=0.5, help='Start timestep for inference')
    args = parser.parse_args()

    config =  {
    'dim': 1536,         
    'depth': 24,         
    'dim_head': 64,      
    'heads': 16,         
    'dropout': 0.1,      
    'ff_expansion_factor': 4, 
    'attn_kwargs': {},  
    'ff_kwargs': {},
    'model_name':args.model_name
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    accelerator = Accelerator(mixed_precision='bf16' if torch.cuda.is_available() else None)
    
    transfusion = Transfusion(num_text_tokens=tokenizer.vocab_size, transformer=config, diffusion_loss_weight=5, flattened_dim=4*args.patch_size*args.patch_size)

    model = ModelWrapper(transfusion, image_size=args.image_size, embed_dim=transfusion.transformer.dim, patch_size=args.patch_size, max_length=args.max_length, vae=vae).to(accelerator.device)

    model, _, _, _, _ = load_checkpoint(args.checkpoint_name, model, None, None)
    model.eval()
    # Load prompt from txt file
    with open('prompt.txt', 'r') as file:
        prompt = file.read().strip()
    print(prompt)
    text = encode_text(prompt, tokenizer, model, max_length=args.max_length)['input_ids'].to(accelerator.device)
    if args.timestep == 0.0:
        latents = torch.randn(1, 4, args.image_size // 8, args.image_size // 8).to(accelerator.device)
    else:
        image = Image.open("example.jpg").convert("RGB")
        latents = vae_encode(image, (args.image_size, args.image_size), vae, accelerator).unsqueeze(0)  
    os.makedirs('generations', exist_ok=True)
    inference(model, vae, None, text, latents, f'generations/gen_{time.time()}.png', args.steps, args.timestep)