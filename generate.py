import argparse
import os

import torch
from data import encode_text, load_checkpoint, patchify, unpatchify
from transfusion import Transfusion
from transformers import AutoTokenizer
from diffusers import AutoencoderKL
from accelerate import Accelerator
from PIL import Image

from vae import to_tensor, vae_decode, vae_encode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the Transfusion model')
    parser.add_argument('--model_name', type=str, default='HuggingFaceTB/SmolLM-1.7B', help='Name of the model to use')
    parser.add_argument('--image_size', type=int, default=256, help='Image size for training')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size for training')
    parser.add_argument('--max_length', type=int, default=128, help='Max length for the input text')
    parser.add_argument('--inference_steps', type=int, default=50, help='Inference steps')
    parser.add_argument('--cfg', type=int, default=1, help='CFG scale')

    args = parser.parse_args()
    model_name = args.model_name

    accelerator = Accelerator()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    vae_name = "madebyollin/sdxl-vae-fp16-fix"
    vae = AutoencoderKL.from_pretrained(vae_name).to(accelerator.device)
    vae.requires_grad = False
    max_length = args.max_length
    image_size = args.image_size
    patch_size = args.patch_size
    inference_steps = args.inference_steps

    model = Transfusion(
    num_text_tokens = tokenizer.vocab_size,
    diffusion_loss_weight=5,
    dim_latent = 4*patch_size*patch_size,
    transformer = {
        'dim': 1536,         
        'depth': 16,         
        'dim_head': 64,      
        'heads': 12,         
        'dropout': 0.1,      
        'ff_expansion_factor': 4,
        'gradient_checkpointing': False,
        'pretrained_model': None
    })

    load_checkpoint("checkpoints/model.pth", model, None, None)
    model = model.to(accelerator.device)

    image_seq_len=(image_size // (patch_size * 8)) ** 2

    with open('prompt.txt', 'r') as file:
        prompt = file.read().strip()
    text = encode_text(prompt, tokenizer, max_length)
    text = torch.cat([
        torch.tensor([model.sos_id]),
        text.input_ids.squeeze(),
        torch.tensor([model.som_ids[0]])
    ]).to(accelerator.device)

    image = Image.open("example.jpg").convert("RGB")    
    latent = vae_encode(image, (image_size, image_size), vae, accelerator).unsqueeze(0)
    latent = patchify(latent, patch_size).squeeze(0)
    noise_percentage = 0.3

    noise = torch.randn_like(latent)
    latent = latent * (1 - noise_percentage) + noise * noise_percentage

    denoised_tokens = unpatchify(latent.detach().clone(), patch_size, 1, 4, image_size // 8, image_size // 8)
    decoded_images = vae_decode(denoised_tokens, vae)
    decoded_images.save("original.png")

    inference_steps = 10

    os.makedirs('inference_results', exist_ok=True)

    modality_sample = model.sample(prompt=text
                                   , initial_latent=latent.clone()
                                   , max_length=len(text)+image_seq_len
                                   , modality_steps=inference_steps
                                   , modality_length=image_seq_len)

    denoised_tokens = unpatchify(modality_sample[1][1], patch_size, 1, 4, image_size // 8, image_size // 8)
    decoded_images = vae_decode(denoised_tokens, vae)

    # Create a folder to save the images if it doesn't exist
    

    # Save the decoded image
    decoded_images.save("image.png")