import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)


def gen_image_by_prompt(prompt, dir, img_name):
    image = pipe(prompt).images[0]
    img_name = f"{dir}/{img_name}.png"
    print(f"Saving img as: {img_name}")
    image.save(img_name)