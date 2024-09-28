"""
Multiscale autoregressive image generation

- Accepts a list of text input_ids and a list of image latents of: 2x2, 4x4, 8x8, 16x16, 32x32, 64x64
- Concatenates the text and the flattened latents into a single tensor
- Passes the concatenated tensor through a transformer decoder
- Unflattens the results into images of each scale
- Predicts each scale conditioned on all previous scales (but not any other pixels of the same/future scales)
- Creates a 4D attention mask to implement this
- Uses two kinds of learned positional embeddings: 
-- lvl_embed (for each scale) 
-- pos_1LC (for each patch in each scale)
- Accepts a pretrained Hugging FaceTransformer as the decoder
- Computes loss on all scale outputs jointly compared to the latent inputs
- Implements a forward method that takes a batch of text and latents and returns a batch of generated latents
- Implements a training loop using the forward method with a dataloader
- Implements an inference method that starts from the text and generates each scale sequentially using the forward method
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vae import vae_decode

class MultiscaleAutoregressiveModel(nn.Module):
    def __init__(self, transformer_decoder, num_scales=6, scale_sizes=[2, 4, 8, 16, 32, 64], latent_dim=4):
        super(MultiscaleAutoregressiveModel, self).__init__()
        self.transformer_decoder = transformer_decoder
        self.num_scales = len(scale_sizes)
        self.scale_sizes = scale_sizes
        self.latent_dim = latent_dim

        # Get transformer hidden dimension from the transformer's config
        transformer_hidden_dim = transformer_decoder.config.hidden_size

        # Linear projections to map between latent_dim and transformer_hidden_dim
        self.latent_to_hidden = nn.Linear(latent_dim, transformer_hidden_dim)
        self.hidden_to_latent = nn.Linear(transformer_hidden_dim, latent_dim)

        # Learned patch positional embeddings
        total_patches = sum([size * size for size in scale_sizes])
        self.pos_1LC = nn.Parameter(torch.zeros(1, total_patches, transformer_hidden_dim))

        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(num_scales, transformer_hidden_dim)
        init_std = math.sqrt(1 / transformer_hidden_dim / 3)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        pos_1LC = []
        for i, pn in enumerate(self.scale_sizes):
            pe = torch.empty(1, pn*pn, transformer_hidden_dim)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)
        self.pos_1LC = nn.Parameter(pos_1LC)

        # Output projection for each scale
        self.output_proj = nn.Linear(transformer_hidden_dim, latent_dim)

    def forward(self, text_input_ids, image_latents):
        """
        Args:
            text_input_ids (Tensor): Shape (batch_size, seq_length)
            image_latents (List[Tensor]): List of Tensors for each scale, each of shape (batch_size, latent_dim, H, W)
        
        Returns:
            generated_latents (List[Tensor]): Generated latents for each scale
        """
        batch_size = text_input_ids.size(0)
        
        # Embed text input_ids
        text_embeddings = self.transformer_decoder.model.embed_tokens(text_input_ids)  # Shape: (batch_size, seq_length, hidden_dim)

        # Flatten and concatenate latents
        flattened_latents = []
        for idx, latents in enumerate(image_latents):
            flattened = latents.view(batch_size, self.latent_dim, -1).permute(0, 2, 1)  # (batch_size, num_patches, latent_dim)
            flattened_latents.append(flattened)
        concatenated_latents = torch.cat(flattened_latents, dim=1)  # (batch_size, total_patches, latent_dim)

        # Project latents to transformer hidden dimension
        transformed_latents = self.latent_to_hidden(concatenated_latents)  # (batch_size, total_patches, hidden_dim)
 
        # Add positional embeddings
        scale_indices = torch.cat([
            torch.full((self.scale_sizes[idx]**2,), idx, dtype=torch.long, device=concatenated_latents.device)
            for idx, latents in enumerate(image_latents)
        ], dim=0)
        scale_embeddings = self.lvl_embed(scale_indices)  # (total_patches, hidden_dim)
        scale_embeddings = scale_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, total_patches, hidden_dim)
        pos_embeddings = self.pos_1LC[:, :transformed_latents.size(1), :]  # (1, total_patches, hidden_dim)
        transformed_latents = transformed_latents + scale_embeddings + pos_embeddings  # (batch_size, total_patches, hidden_dim)

        # Concatenate text embeddings and transformed image latents
        combined = torch.cat([text_embeddings, transformed_latents], dim=1)  # (batch_size, seq_length + total_patches, hidden_dim)

        # Create attention mask
        attention_mask = self.create_attention_mask(self.num_scales, self.scale_sizes, combined.size(1), text_len=text_embeddings.size(1))
                                                                                                                                                    # Convert 2D mask to 4D
        attention_mask = attention_mask.to("mps").unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        attention_mask = attention_mask.expand(batch_size, 1, combined.size(1), combined.size(1))  # Expand to batch size

        # Pass through transformer decoder
        decoder_output = self.transformer_decoder(inputs_embeds=combined, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]  # (batch_size, seq_length + total_patches, hidden_dim)

        # Split decoder output back into scales
        generated_latents = []
        start_idx = text_embeddings.size(1)
        text_predictions = self.transformer_decoder.lm_head(decoder_output[:, :start_idx, :])
        for i in range(self.num_scales):
            end_idx = start_idx + self.scale_sizes[i] * self.scale_sizes[i]
            tokens = decoder_output[:, start_idx:end_idx, :]
            latent = self.output_proj(tokens)  # (batch_size, num_patches, size*size)
            latent = latent.view(batch_size, 4, self.scale_sizes[i], self.scale_sizes[i])  # (batch_size, latent_dim, H, W)
            generated_latents.append(latent)
            start_idx = end_idx

        return generated_latents, text_predictions

    def create_attention_mask(self, num_scales, scale_sizes, total_length, text_len):
        """
        Creates a causal attention mask where each scale can attend to all previous scales and text, but not the current or future scales.

        Args:
            num_scales (int): Number of scales
            scale_sizes (List[int]): List of spatial sizes for each scale
            total_length (int): Total sequence length (text + all patches)
            text_len (int): Length of the text sequence

        Returns:
            mask (Tensor): Boolean mask of shape (total_length, total_length)
        """
        mask = torch.ones((total_length, total_length), dtype=torch.bool)
        current = text_len
        """
        for i, size in enumerate(scale_sizes):
            num_patches = size * size
            # Allow attention to text and all previous scales
            mask[current:current+num_patches, :current] = True
            # Prevent attention to current and future scales
            mask[current:current+num_patches, current:] = False
            current += num_patches
        """

        #self.visualize_attention_mask(mask, text_len, scale_sizes, "attention_mask.png")
        
        return mask

    def visualize_attention_mask(self, mask, text_len, scale_sizes, output_path):
        """
        Visualizes the attention mask and saves it as a PNG file.

        Args:
            num_scales (int): Number of scales
            scale_sizes (List[int]): List of spatial sizes for each scale
            total_length (int): Total sequence length (text + all patches)
            text_len (int): Length of the text sequence
            output_path (str): Path to save the output PNG file
        """
        mask_np = mask.cpu().numpy()

        plt.figure(figsize=(12, 12))
        plt.imshow(mask_np, cmap='binary', interpolation='nearest')
        plt.colorbar()

        # Add lines to separate text and different scales
        current = text_len
        plt.axhline(y=text_len - 0.5, color='r', linestyle='-', linewidth=2)
        plt.axvline(x=text_len - 0.5, color='r', linestyle='-', linewidth=2)
        for size in scale_sizes:
            current += size * size
            plt.axhline(y=current - 0.5, color='r', linestyle='-', linewidth=2)
            plt.axvline(x=current - 0.5, color='r', linestyle='-', linewidth=2)

        plt.title("Attention Mask Visualization")
        plt.xlabel("Key")
        plt.ylabel("Query")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Attention mask visualization saved to {output_path}")

    def compute_loss(self, generated_latents, target_latents, text_logits, text_labels, weighting_factor=10):
        """
        Computes the joint loss for all scale outputs and autoregressive text loss.

        Args:
            generated_latents (List[Tensor]): Generated latents for each scale
            target_latents (List[Tensor]): Target latents for each scale
            text_logits (Tensor): Predicted text logits
            text_labels (Tensor): Ground truth text labels

        Returns:
            loss (Tensor): Scalar loss value
        """
        # Compute latent loss
        latent_loss = 0
        for gen, target in zip(generated_latents, target_latents):
            latent_loss += F.mse_loss(gen, target)
        
        # Compute autoregressive text loss
        text_loss = F.cross_entropy(text_logits.view(-1, text_logits.size(-1)), text_labels.view(-1), ignore_index=-100)

        total_loss = text_loss + weighting_factor * latent_loss
        
        return total_loss,text_loss, latent_loss

    def inference(self, text_input_ids, device):
        """
        Generates image latents sequentially for each scale based on the input text.

        Args:
            text_input_ids (Tensor): Shape (batch_size, seq_length)
            device (torch.device): Device to run the inference on

        Returns:
            generated_latents (List[Tensor]): Generated latents for each scale
        """
        self.eval()
        with torch.no_grad():
            batch_size = text_input_ids.size(0)
            generated_latents = []
            for i in range(self.num_scales):
                current_latents = generated_latents.copy()
                latents = self.forward(text_input_ids, current_latents)
                generated_latents = latents  # Update with the newly generated latents
            return generated_latents
        
    

