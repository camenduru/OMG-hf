import spaces
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(...)
pipe.to('cuda')

@spaces.GPU
def generate(prompt):
    return pipe(prompt).images

gr.Interface(
    fn=generate,
    inputs=gr.Text(),
    outputs=gr.Gallery(),
).launch()