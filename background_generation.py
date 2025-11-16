import logging
import os
from enum import Enum
from pathlib import Path

import requests

from scene_classes import Scene

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BackgroundModel(str, Enum):
    """
    Indicates the model to use for background generation.
    """

    STABLE_DIFFUSION = "stable_diffusion"
    HUNYUAN_PANORAMA = "hunyuan_panorama"
    DIFFUSION360 = "diffusion360"


def generate_stable_diffusion_background(
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
    model: BackgroundModel = BackgroundModel.STABLE_DIFFUSION,
    server_url: str | None = None,
) -> None:
    """
    Generate a background image for the scene in the output directory
    Args:
        scene (Scene): The scene containing objects to generate
        output_dir (Path): The directory to save the generated GLB files
        model (BackgroundModel): The model to use for background generation
        server_url (str | None): The URL of the server for remote models
    Returns:
        None
    """

    if model == BackgroundModel.HUNYUAN_PANORAMA:
        if server_url is None:
            raise ValueError("server_url is required for HUNYUAN_PANORAMA model")

        # Call the HunyuanWorld panorama generation model
        object_request = requests.get(
            f"{server_url}/generate-panorama",
            params={
                "prompt": scene.scene_skybox_prompt,
                "negative_prompt": scene.scene_skybox_negative_prompt,
            },
        )

        background_file_path = output_dir / "background.png"
        with open(background_file_path, "wb") as f:
            f.write(object_request.content)

    elif model == BackgroundModel.DIFFUSION360:
        if server_url is None:
            raise ValueError("server_url is required for DIFFUSION360 model")

        # Call the Diffusion360 panorama generation model
        object_request = requests.post(
            f"{server_url}/generate/file",
            json={
                "prompt": scene.scene_skybox_prompt,
                "negative_prompt": scene.scene_skybox_negative_prompt,
            },
        )

        background_file_path = output_dir / "background.png"
        with open(background_file_path, "wb") as f:
            f.write(object_request.content)

    elif model == BackgroundModel.STABLE_DIFFUSION:
        # Generate the background using Stable Diffusion locally
        generate_stable_diffusion_background(scene, output_dir)

    else:
        raise ValueError(f"Unknown background model: {model}")
