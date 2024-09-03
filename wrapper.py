from einops import rearrange
import torch
import torch.nn.functional as F

from transfusion import LossBreakdown

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, vae, image_size=256, max_length=256, embed_dim=512, patch_size=2):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.image_size = image_size
        self.latent_dim = image_size // (patch_size * 8)
        self.patch_size = patch_size
        self.max_length = max_length
        self.vae = vae

    def forward(self, text, latents, times=None, return_loss=True, num_inference_steps=50, start_timestep=0.0):
        # Patchify and flatten the latents
        B, C, H, W = latents.shape # 1, 4, 32, 32
        latents = latents.view(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        latents = latents.permute(0, 2, 4, 1, 3, 5).contiguous()
        latents = latents.view(B, -1, C * self.patch_size * self.patch_size)
        
        modality_tokens = [[latents[i].unsqueeze(0)] for i in range(B)]

        # Calculate the number of text tokens
        num_text_tokens = self.max_length
        
        # Calculate the number of image tokens (latent patches)
        num_image_tokens = self.latent_dim * self.latent_dim

        modality_positions = [[(num_text_tokens+1, num_image_tokens)] for _ in range(B)]

        def unpatchify(x):
            x = x.view(B, self.latent_dim, self.latent_dim, C, self.patch_size, self.patch_size)
            x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
            return x.view(B, C, H, W)

        loss, loss_dict, denoised_tokens, noise, flow, pred_flow, noised_image = self.model(text=text, modality_tokens=modality_tokens, modality_positions=modality_positions, times=times, return_loss=return_loss, num_inference_steps=num_inference_steps, start_timestep=start_timestep)

        # Unpatchify all relevant tensors
        unpatchified = unpatchify(denoised_tokens)
        noise = unpatchify(noise)
        flow = unpatchify(flow) if flow is not None else None
        pred_flow = unpatchify(pred_flow)
        noised_image = unpatchify(noised_image) if noised_image is not None else None

        return loss, loss_dict, unpatchified, noise, flow, pred_flow, noised_image