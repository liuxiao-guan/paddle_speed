import paddle

# from ppdiffusers import FluxPipeline
from noiseFluxpipeline import NoiseFluxPipeline

pipe = NoiseFluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16 ,
)

prompt = "An image of a squirrel in Picasso style"
for i in range(2):
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=paddle.Generator().manual_seed(44)
    ).images[0]
    image.save("text_to_image_generation-flux-dev-result.png")