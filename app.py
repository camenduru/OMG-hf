import spaces
from diffusers import DiffusionPipeline
import torch
import gradio as gr

pipe = DiffusionPipeline.from_pretrained("Fucius/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16")
pipe.to('cuda')

@spaces.GPU
def generate(prompt):
    return pipe(prompt).images

gr.Interface(
    fn=generate,
    inputs=gr.Text(),
    outputs=gr.Gallery(),
).launch()