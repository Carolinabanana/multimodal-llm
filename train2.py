from torch import randint, randn
import torch
from transfusion import Transfusion

model = Transfusion(
    num_text_tokens = 256,
    dim_latent = 384,
    transformer = dict(
        dim = 512,
        depth = 8
    )
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

"""
# Generate the data
text_and_images = [
    [randint(0, 256, (16,)), randn(4, 384), randint(0, 256, (8,)), randn(6, 384)],
    [randint(0, 256, (16,)), randn(7, 384), randint(0, 256, (5,)), randn(2, 384), randint(0, 256, (9,))]
]

# Save the data
torch.save(text_and_images, 'text_and_images.pt')
"""
# Load the data


for i in range(100):
    optimizer.zero_grad()
    text_and_images = torch.load('text_and_images.pt')
    loss = model(text_and_images)

    print(loss)

    loss.backward()
    optimizer.step()