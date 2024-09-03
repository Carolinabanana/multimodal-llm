import os
import torch
from data import TransfusionDataset, create_text_image_pairs, load_checkpoint, load_pairs_from_disk, resume_checkpoint, save_checkpoint, save_pairs_to_disk
from transfusion import Transfusion
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from vae import vae_decode
from wrapper import ModelWrapper
import argparse
from tqdm import tqdm
from inference import debug_image, inference
from schedulefree import AdamWScheduleFree
from accelerate import Accelerator

def train():
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description='Train the Transfusion model')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--model_name', type=str, default='HuggingFaceTB/SmolLM-1.7B', help='Name of the model to use')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=256, help='Image size for training')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size for training')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Use gradient checkpointing')
    parser.add_argument('--diffusion_loss_weight', type=float, default=5, help='Weight for the diffusion loss')
    parser.add_argument('--max_length', type=int, default=64, help='Max length for the input text')
    parser.add_argument('--debug_steps', type=int, default=100, help='Number of steps to debug')
    parser.add_argument('--inference_steps', type=int, default=200, help='Number of steps to inference')
    parser.add_argument('--save_steps', type=int, default=1000, help='Number of steps to save')

    args = parser.parse_args()
    model_name = args.model_name

    accelerator = Accelerator(mixed_precision='bf16' if torch.cuda.is_available() else None)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    max_length = args.max_length
    image_size = args.image_size
    batch_size = args.batch_size
    patch_size = args.patch_size
    learning_rate = args.learning_rate
    diffusion_loss_weight = args.diffusion_loss_weight
    warmup_steps = 0
    N_debug = args.debug_steps
    N_inference = args.inference_steps
    N_save = args.save_steps
    num_epochs = 10
    N_loss_window = 100  # Number of steps for moving average
    device = accelerator.device
    gradient_checkpointing = args.gradient_checkpointing

    # Update the model instantiation
    config =  {
    'dim': 1536,         
    'depth': 24,         
    'dim_head': 64,      
    'heads': 16,         
    'dropout': 0.1,      
    'ff_expansion_factor': 4, 
    'attn_kwargs': {},  
    'ff_kwargs': {},
    'model_name':model_name,
    'gradient_checkpointing': gradient_checkpointing
}
    
    
    print(f"Model name: {model_name}")
    print(f"Learning rate: {learning_rate}")
    print(f"Gradient checkpointing: {gradient_checkpointing}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}")
    print(f"Patch size: {patch_size}")
    print(f"Diffusion loss weight: {diffusion_loss_weight}")
    print(f"Debug: {N_debug} Inference: {N_inference} Save: {N_save}")
    print(f"Max length: {max_length}")

    transfusion = Transfusion(num_text_tokens=tokenizer.vocab_size, transformer=config, diffusion_loss_weight=diffusion_loss_weight, flattened_dim=vae.config.latent_channels*patch_size*patch_size)

    model = ModelWrapper(transfusion, image_size=image_size, embed_dim=transfusion.transformer.dim, patch_size=patch_size, max_length=max_length, vae=vae).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")
    


    if not os.path.isfile("pairs.pkl"):
        pairs = create_text_image_pairs("source")
        save_pairs_to_disk(pairs,"pairs.pkl")
    text_image_pairs = load_pairs_from_disk('pairs.pkl')
    print(f"Loaded {len(text_image_pairs)} text-image pairs")

    dataset = TransfusionDataset(text_image_pairs, tokenizer, model, max_length=max_length, image_size=image_size, device=device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = AdamWScheduleFree(model.parameters(), lr=learning_rate, foreach=torch.cuda.is_available(), warmup_steps=warmup_steps)
    optimizer.train()

    # Add LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader) / batch_size, eta_min=1e-6)

    # Initialize epoch and step counter
    start_epoch = 0
    step_counter = 0

    # Load checkpoint if resume flag is set
    if args.resume:
        model, optimizer, scheduler, start_epoch, step_counter = resume_checkpoint(model, optimizer, scheduler, device)

    # Prepare everything with accelerator
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    # Sample for inference
    sample_text = next(iter(dataloader))['input_ids'][0].unsqueeze(0).to(device)
    sample_latents = next(iter(dataloader))['image_latents'][0].unsqueeze(0).to(device)

    # Decode the sample latents using the VAE
    decoded_sample_latents = vae_decode(sample_latents, model.vae)

    # Create a folder to save the decoded sample latents if it doesn't exist
    os.makedirs('inference_results', exist_ok=True)
    decoded_sample_latents.save(f'inference_results/sample_latents_epoch_0.png')
    
    # Add noise to the sample latents
    noise = torch.randn_like(sample_latents)
    sample_latents = 0.5 * sample_latents + 0.5 * noise

    text_loss_window = []
    diffusion_loss_window = []

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        epoch_text_loss = 0
        epoch_diffusion_loss = 0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/10", leave=False)
        for i, batch in enumerate(progress_bar, 1):
            step_counter += 1
            optimizer.zero_grad()
            
            text = batch['input_ids'].to(device)
            latents = batch['image_latents'].to(device)

            times = torch.rand((latents.shape[0], 1), device=device)

            if step_counter % N_inference == 0 or step_counter % N_debug == 0:
                timestep = 0.7 
                times = torch.full((latents.shape[0],1), timestep, device=latents.device)
            
            with accelerator.autocast():
                loss, loss_dict, denoised_tokens, noise, flow, pred_flow, noised_image = model(text=text, latents=latents, times=times, return_loss=True)
            
            epoch_text_loss += loss_dict.text.item()
            epoch_diffusion_loss += loss_dict.diffusion.item()
            
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            
            # Update loss windows
            text_loss_window.append(loss_dict.text.item())
            diffusion_loss_window.append(loss_dict.diffusion.item())
            
            if len(text_loss_window) > N_loss_window:
                text_loss_window.pop(0)
            if len(diffusion_loss_window) > N_loss_window:
                diffusion_loss_window.pop(0)
            
            # Calculate moving averages
            avg_text_loss = sum(text_loss_window) / len(text_loss_window)
            avg_diffusion_loss = sum(diffusion_loss_window) / len(diffusion_loss_window)
            
            progress_bar.set_postfix({
                'avg_text_loss': f"{avg_text_loss:.4f}",
                'avg_diffusion_loss': f"{avg_diffusion_loss:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}"
            })

            if step_counter % N_debug == 0:
                debug_image(model, latents, noise, pred_flow, flow, noised_image, denoised_tokens, epoch, step_counter)

            
            if step_counter % N_inference == 0:
                inference(model, optimizer, sample_text, sample_latents, f'inference_results/inference_epoch_{epoch+1}_step_{step_counter}.png')

            # Save the model every N steps
            if step_counter % N_save == 0:
                save_checkpoint(model, optimizer, scheduler, loss, epoch, step_counter)
        
if __name__ == '__main__':
    train() 

