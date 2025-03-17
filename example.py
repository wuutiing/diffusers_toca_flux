

import torch

from diffusers import FluxPipeline
from toca_impl.toca import TokenCacheObj

model_path = "black-forest-labs/FLUX.1-dev"
pipeline = FluxPipeline.from_pretrained(model_path,
                                        torch_dtype=torch.bfloat16).to("cuda")

def inference(prompt="a cat.", steps=28, seed=19):
    token_cache_obj = TokenCacheObj(steps)
    image = pipeline(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=steps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed),
        joint_attention_kwargs={"token_cache_obj": token_cache_obj}
    ).images[0]
    return image

if __name__ == "__main__":
    image = inference()
    image.save("output.png")

