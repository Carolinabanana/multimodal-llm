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
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the Transfusion model')
    parser.add_argument('--checkpoint_name', type=str, default='model_checkpoint_epoch_5_step_5000.pth', help='Load checkpoint')
    parser.add_argument('--model_name', type=str, default='HuggingFaceTB/SmolLM-1.7B', help='Name of the model to use')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=256, help='Image size for training')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size for training')
    parser.add_argument('--max_length', type=int, default=64, help='Max length for text')
    parser.add_argument('--steps', type=int, default=50, help='Number of steps for inference')
    parser.add_argument('--timestep', type=float, default=0.8, help='Start timestep for inference')
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
    device = "cuda" if torch.cuda.is_available() else "mps"
    
    transfusion = Transfusion(num_text_tokens=tokenizer.vocab_size, transformer=config, diffusion_loss_weight=5, flattened_dim=vae.config.latent_channels*args.patch_size*args.patch_size)

    model = ModelWrapper(transfusion, image_size=args.image_size, embed_dim=transfusion.transformer.dim, patch_size=args.patch_size, max_length=args.max_length, vae=vae).to(device)

    model, _, _, _, _ = load_checkpoint(args.checkpoint_name, model, None, None, device)
    model.eval()
    prompt = "a beautiful landscape with a river and a mountain"
    text = encode_text(prompt, tokenizer, model, max_length=args.max_length)['input_ids'].to(device)
    if args.timestep == 0.0:
        latents = torch.randn(1, 4, args.image_size // 8, args.image_size // 8).to(device)
    else:
        image = Image.open("example.jpg").convert("RGB")
        latents = vae_encode(image, (args.image_size, args.image_size), vae).unsqueeze(0)  
    os.makedirs('generations', exist_ok=True)
    inference(model, None, text, latents, f'generations/gen_{time.time()}.png', args.steps, args.timestep)