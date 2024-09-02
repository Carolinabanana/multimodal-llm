import os
import torch
from data import TransfusionDataset, load_pairs_from_disk, save_checkpoint
from transfusionpytorch.transfusion_pytorch.transfusion import Transfusion
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from wrapper import ModelWrapper
import argparse
from tqdm import tqdm
from inference import debug_image, inference

def train():
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description='Train the Transfusion model')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()

    text_image_pairs = load_pairs_from_disk('pairs.pkl')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    max_length = 64
    image_size = 256
    batch_size = 8
    patch_size = 2
    N_debug = 10
    N_inference = 10
    N_save = 1000
    N_loss_window = 100  # Number of steps for moving average
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    # Update the model instantiation
    config =  {
    'dim': 512,         
    'depth': 12,         
    'dim_head': 64,      
    'heads': 16,         
    'dropout': 0.1,      
    'ff_expansion_factor': 4, 
    'attn_kwargs': {},  
    'ff_kwargs': {},
    'model_name': None
}
    transfusion = Transfusion(num_text_tokens=tokenizer.vocab_size+3, transformer=config, diffusion_loss_weight=5, flattened_dim=vae.config.latent_channels*patch_size*patch_size)
    # Print model parameter count
    
    model = ModelWrapper(transfusion, image_size=image_size, embed_dim=transfusion.transformer.dim, patch_size=patch_size, max_length=max_length, vae=vae).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")

    dataset = TransfusionDataset(text_image_pairs, tokenizer, model, max_length=max_length, image_size=image_size, device=device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Initialize epoch and step counter
    start_epoch = 0
    step_counter = 0

    # Load checkpoint if resume flag is set
    if args.resume:
        checkpoint_files = [f for f in os.listdir('./checkpoints') if f.startswith('model_checkpoint_epoch_') and f.endswith('.pth')]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files)
            print(f"Loading checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(f'./checkpoints/{latest_checkpoint}')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            step_counter = checkpoint['step_counter']
            print(f"Resuming from epoch {start_epoch}")
        else:
            print("No checkpoint found. Starting from scratch.")

    # Sample for inference
    sample_text = next(iter(dataloader))['input_ids'][0].unsqueeze(0).to(device)
    sample_latents = next(iter(dataloader))['image_latents'][0].unsqueeze(0).to(device)
    
    # Add noise to the sample latents
    noise = torch.randn_like(sample_latents)
    sample_latents = 0.5 * sample_latents + 0.5 * noise

    text_loss_window = []
    diffusion_loss_window = []

    for epoch in range(start_epoch, 10):
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
            
            loss, loss_dict, denoised_tokens, noise, flow, pred_flow, noised_image = model(text=text, latents=latents, times=times, return_loss=True)
            
            epoch_text_loss += loss_dict.text.item()
            epoch_diffusion_loss += loss_dict.diffusion.item()
            
            loss.backward()
            optimizer.step()
            
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
                'avg_diffusion_loss': f"{avg_diffusion_loss:.4f}"
            })

            if step_counter % N_debug == 0:
                debug_image(model, latents, noise, pred_flow, flow, noised_image, denoised_tokens, epoch, step_counter)

            
            if step_counter % N_inference == 0:
                inference(model, sample_text, sample_latents, epoch, step_counter)

            # Save the model every N steps
            if step_counter % N_save == 0:
                save_checkpoint(model, optimizer, loss, epoch, step_counter)
        
if __name__ == '__main__':
    train() 






