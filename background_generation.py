import logging
import os
from pathlib import Path

import requests

from scene_classes import Scene

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_background_image(
    scene: Scene,
    output_dir: Path = Path("./Unity/AIML Research Project/Assets"),
) -> None:
    """
    Generate a background image using the skybox prompt from scene.
    Args:
        scene (Scene): The scene object containing the skybox prompt.
        output_dir (Path): The directory to save the background image in.
    Returns:
        None
    """

    logger.debug(f"generate_background called with output_dir: {output_dir}")
    import torch
    from diffusers import StableDiffusionPipeline

    logger.debug("Loading Stable Diffusion pipeline")
    pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16)

    pipeline = pipeline.to("cuda")
    logger.debug("Pipeline loaded and moved to CUDA")

    prompt = scene.scene_skybox_prompt
    logger.debug(f"Generating background with prompt: {prompt}")

    images = pipeline(prompt=prompt, height=2048, width=2048, num_inference_steps=50, guidance_scale=7.5).images
    image = images[0]
    background_file_path = output_dir / "background.png"
    logger.debug(f"Saving background image to: {background_file_path}")
    image.save(background_file_path)
    logger.debug("Background image successfully generated and saved")


def generate_background_image(
    scene: Scene,
    output_dir: Path = Path("./Unity/AIML Research Project/Assets"),
    hunyuan_panorama_server_url: str | None = None,
) -> None:
    """
    Generate a background image for the scene in the output directory
    Args:
        scene (Scene): The scene containing objects to generate
        output_dir (Path): The directory to save the generated GLB files
        hunyuan_panorama_server_url (str | None): The URL of the Hunyuan panorama server, if used
    Returns:
        None
    """

    if hunyuan_panorama_server_url is None:
        save_background_image(scene, output_dir)
    else:
        object_request = requests.get(
            f"{hunyuan_panorama_server_url}/generate-panorama",
            params={
                "prompt": scene.scene_skybox_prompt,
                "negative_prompt": scene.scene_skybox_negative_prompt,
            },
        )

        background_file_path = output_dir / "background.png"
        with open(background_file_path, "wb") as f:
            f.write(object_request.content)
