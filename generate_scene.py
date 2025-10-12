import json
import logging
import os
import warnings
from pathlib import Path

import typer

from scene_classes import Scene
from model_generation import generate_object_models

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Hide future warnings from shap_e
warnings.filterwarnings("ignore", category=FutureWarning)


def generate_scene_object(prompt: str) -> Scene:
    """
    Generate a scene layout using an LLM based on the provided prompt.
    Args:
        prompt (str): The text prompt describing the desired scene.
    Returns:
        Scene: The generated scene layout.
    """

    logger.debug(f"generate_scene_object called with prompt: {prompt}")
    from openai import OpenAI

    openai_client = OpenAI()
    logger.debug("Generating scene with LLM")
    response = openai_client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {
                "role": "system",
                "content": (
                    "Imagine a scene based on the prompt with many objects in the scene."
                    "Extract the scene information. Ensure the prompt for the scene skybox could generate a skybox"
                    "image that could be used on a spherical shader. Add as many objects to the scene as needed,"
                    "multiple copies of the same object are cheap and easy to do. Include a camera object with position"
                    "and rotation_euler_angles_degrees to set the camera position and angle so that it captures the"
                    "objects in the scene."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        text_format=Scene,
    )

    scene = response.output_parsed
    if scene is None:
        logger.error("No scene generated from LLM")
        raise Exception("No scene generated")

    logger.debug(f"Successfully generated scene with {len(scene.objects)} objects")
    return scene


def save_scene_json(scene: Scene, output_dir: Path = Path("./Unity/AIML Research Project/Assets")) -> None:
    """
    Save the scene layout to a JSON file in the specified output directory.
    Args:
        scene (Scene): The scene layout to save.
        output_dir (Path): The directory to save the JSON file in.
    Returns:
        None
    """

    logger.debug(f"save_scene called with output_dir: {output_dir}")
    scene_json = scene.model_dump_json(indent=4)
    scene_file_path = output_dir / "scene.json"
    logger.debug(f"Writing scene to file: {scene_file_path}")
    with open(scene_file_path, "w") as scene_file:
        scene_file.write(scene_json)
    logger.debug("Scene successfully saved to file")


def save_background_image(scene: Scene, output_dir: Path = Path("./Unity/AIML Research Project/Assets")) -> None:
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

    prompt = scene.stable_diffusion_scene_skybox_prompt
    logger.debug(f"Generating background with prompt: {prompt}")

    images = pipeline(prompt=prompt, height=2048, width=2048, num_inference_steps=50, guidance_scale=7.5).images
    image = images[0]
    background_file_path = output_dir / "background.png"
    logger.debug(f"Saving background image to: {background_file_path}")
    image.save(background_file_path)
    logger.debug("Background image successfully generated and saved")


app = typer.Typer(add_completion=False)


@app.command()
def generate_scene(
    prompt: str,
    hunyuan_server_url: str | None = None,
    output_dir: Path = Path("./Unity/AIML Research Project/Assets"),
    model_batch_size: int = 1,
    model_guidance_scale: float = 15.0,
) -> None:
    """
    Generate a complete scene including layout, object models, and background image.

    Args:
        prompt (str): The text prompt describing the desired scene.
        hunyuan_server_url (str | None): The URL of the Hunyuan server, if used.
        output_dir (Path): The directory to save generated assets.
        model_batch_size (int): The batch size for model generation.
        model_guidance_scale (float): The guidance scale for model generation.
    Returns:
        None
    """

    logger.debug(
        f"generate_scene command called with prompt: {prompt}, "
        f"output_dir: {output_dir}, model_batch_size: {model_batch_size}, "
        f"model_guidance_scale: {model_guidance_scale}"
    )
    scene = generate_scene_object(prompt)
    logger.debug("Scene object generated, saving scene")
    save_scene_json(scene, output_dir)
    logger.debug("Scene saved, generating object models")
    generate_object_models(
        scene,
        output_dir,
        batch_size=model_batch_size,
        guidance_scale=model_guidance_scale,
        hunyuan_server_url=hunyuan_server_url,
    )
    logger.debug("Object models generated, generating background")
    save_background_image(scene, output_dir)
    logger.debug("Scene generation completed successfully")


@app.command()
def generate_scene_layout(prompt: str) -> None:
    """
    Generate a scene layout using an LLM based on the provided prompt and print the JSON.
    Args:
        prompt (str): The text prompt describing the desired scene.
    Returns:
        None
    """
    logger.debug(f"generate_scene_layout called with prompt: {prompt}")
    scene = generate_scene_object(prompt)
    scene_json = scene.model_dump_json(indent=4)
    logger.info(scene_json)
    logger.debug("Scene layout generated")


@app.command(help="Generate GLB models for each object from the existing scene.json.")
def models(
    hunyuan_server_url: str | None = None,
) -> None:
    """
    Read scene.json and generate GLB models for unique object names.
    Args:
        hunyuan_server_url (str | None): The URL of the Hunyuan server, if used.
    Returns:
        None
    """
    logger.debug("models command called")
    scene_file_path = "./Unity/AIML Research Project/Assets/scene.json"
    logger.debug(f"Reading scene from: {scene_file_path}")
    with open(scene_file_path) as f:
        scene_dict = json.load(f)
    s = Scene.model_validate(scene_dict)
    logger.debug(f"Scene loaded with {len(s.objects)} objects")
    generate_object_models(s, hunyuan_server_url=hunyuan_server_url)
    logger.debug("Model generation completed")


@app.command(help="Generate a background image using the skybox prompt from scene.json.")
def background() -> None:
    """Generate the skybox image only (using Stable Diffusion)"""
    logger.debug("background command called")
    scene_file_path = "./Unity/AIML Research Project/Assets/scene.json"
    logger.debug(f"Reading scene from: {scene_file_path}")
    with open(scene_file_path) as f:
        scene_dict = json.load(f)
    s = Scene.model_validate(scene_dict)
    logger.debug("Scene loaded, generating background")
    save_background_image(s)
    logger.debug("Background generation completed")


if __name__ == "__main__":
    app()
