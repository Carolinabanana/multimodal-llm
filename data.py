import os
import pickle
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from vae import vae_encode

class TransfusionDataset(Dataset):
    def __init__(self, text_image_pairs, tokenizer, model, max_length, image_size, device):
        self.text_image_pairs = text_image_pairs
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_length = max_length
        self.image_size = image_size
        
        # Add special tokens
        special_tokens = {"additional_special_tokens": ["<BOI>", "<EOI>","<MODALITY>"]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.text_image_pairs)

    def __getitem__(self, idx):
        text, image_path = self.text_image_pairs[idx]
        # Calculate the length of the image sequence
        image_seq_len = self.model.latent_dim * self.model.latent_dim

        # Tokenize text with special tokens
        tokenized_text = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        # Create the image token sequence
        image_tokens = f"<BOI>"+"<MODALITY>"*image_seq_len+"<EOI>"
        
        # Append image tokens to the tokenized text
        image_token_ids = self.tokenizer.encode(image_tokens, add_special_tokens=False)
        tokenized_text['input_ids'] = torch.cat([tokenized_text['input_ids'], torch.tensor([image_token_ids])], dim=1)
        tokenized_text['attention_mask'] = torch.cat([tokenized_text['attention_mask'], torch.ones(1, len(image_token_ids))], dim=1)

        image = Image.open(image_path).convert("RGB")
        
        # Load and process image
        image_latents = vae_encode(image, (self.image_size, self.image_size), self.model.vae)  
        
        return {
            "input_ids": tokenized_text.input_ids.squeeze(),
            "attention_mask": tokenized_text.attention_mask.squeeze(),
            "image_latents": image_latents
        }
    
def create_text_image_pairs(folder_path):
    text_image_pairs = []
    
    # List all files in the folder
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

def save_checkpoint(model, optimizer, loss, epoch, step_counter):
    print(f"Saving model at step {step_counter}...")
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'step_counter': step_counter,
    }, f'checkpoints/model_checkpoint_epoch_{epoch+1}_step_{step_counter}.pth')
    print("Model saved successfully.")
    