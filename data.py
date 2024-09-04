import os
import pickle
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from vae import to_tensor, vae_encode

class TransfusionDataset(Dataset):
    def __init__(self, text_image_pairs, tokenizer, model, max_length, image_size, device):
        self.text_image_pairs = text_image_pairs
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_length = max_length
        self.image_size = image_size

    def __len__(self):
        return len(self.text_image_pairs)

    def __getitem__(self, idx):
        text, image_path = self.text_image_pairs[idx]

        tokenized_text = encode_text(text, self.tokenizer, self.model, self.max_length)
        # Calculate the length of the image sequence
        
        image = Image.open(image_path).convert("RGB")
        
        # Load and process image
        #image_latents = vae_encode(image, (self.image_size, self.image_size), self.model.vae) 
        pixel_values = to_tensor(image, (self.image_size, self.image_size)) 
        
        return {
            "input_ids": tokenized_text.input_ids.squeeze(),
            "attention_mask": tokenized_text.attention_mask.squeeze(),
            "pixel_values": pixel_values
        }
    
def encode_text(text, tokenizer, model, max_length):
    image_seq_len = model.latent_dim * model.latent_dim

    # Tokenize text with special tokens
    tokenized_text = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

    # Create the image token sequence
    image_tokens = f"<|im_start|>"+"<|endoftext|>"*image_seq_len+"<|im_end|>"
    
    # Append image tokens to the tokenized text
    image_token_ids = tokenizer.encode(image_tokens, add_special_tokens=False)
    tokenized_text['input_ids'] = torch.cat([tokenized_text['input_ids'], torch.tensor([image_token_ids])], dim=1)
    tokenized_text['attention_mask'] = torch.cat([tokenized_text['attention_mask'], torch.ones(1, len(image_token_ids))], dim=1)

    return tokenized_text

    
def create_text_image_pairs(folder_paths):
    text_image_pairs = []
    
    # List all files in the folder
    for folder_path in folder_paths:
        files = os.listdir(folder_path)

        # Iterate through the files
        for file in files:
            # Check if the file is an image (png or jpg)
            if file.lower().endswith(('.png', '.jpg', '.jpeg','.webp')):
                # Get the file name without extension
                base_name = os.path.splitext(file)[0]
                
                # Check if a corresponding text file exists
                txt_file = base_name + '.txt'
                if txt_file in files:
                    # Read the content of the text file
                    with open(os.path.join(folder_path, txt_file), 'r') as f:
                        text_content = f.read().strip()
                    
                    # Create the pair and add it to the list
                    image_path = os.path.join(folder_path, file)
                    text_image_pairs.append((text_content, image_path))
    
    return text_image_pairs

def save_pairs_to_disk(pairs, filename):
    with open(filename, 'wb') as f:
        pickle.dump(pairs, f)

def load_pairs_from_disk(filename):
    with open(filename, 'rb') as f:
        pairs = pickle.load(f)
    return pairs

def save_checkpoint(model, optimizer, scheduler, loss, epoch, step_counter):
    print(f"Saving model at step {step_counter}...")
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        #'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'step_counter': step_counter,
    }, f'checkpoints/model_checkpoint_epoch_{epoch+1}_step_{step_counter}.pth')
    print("Model saved successfully.")

def resume_checkpoint(model, optimizer, scheduler, device):
    checkpoint_files = [f for f in os.listdir('./checkpoints') if f.startswith('model_checkpoint_epoch_') and f.endswith('.pth')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files)
        return load_checkpoint(f'./checkpoints/{latest_checkpoint}', model, optimizer, scheduler, device)

def load_checkpoint(model_path, model, optimizer, scheduler, device):
    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict']) if optimizer is not None else None
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict']) if scheduler is not None else None
    start_epoch = checkpoint['epoch'] + 1
    step_counter = checkpoint['step_counter']
    print(f"Resuming from epoch {start_epoch} step {step_counter}")

    return model, optimizer, scheduler, start_epoch, step_counter

# Create a new dataset that loads from cache
class CachedDataset(Dataset):
    def __init__(self, cache_dir, batch_size):
        self.cache_files = sorted([f for f in os.listdir(cache_dir) if f.startswith(f'batch_{batch_size}_')])
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.cache_dir, self.cache_files[idx]), map_location=torch.device('cpu'))
