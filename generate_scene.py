import json
import logging
import os
import warnings
from pathlib import Path

import typer

from scene_classes import Scene
from model_generation import generate_object_models, slugify
from background_generation import generate_background_image

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_file = Path("scene_generation.log")
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a"),  # Append mode
        logging.StreamHandler(),  # Also output to console
    ],
    force=True,  # Override any existing logging configuration
)
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
                    "image that could be used on a panoramic spherical shader. Include negative prompts in the skybox"
                    "as well to tune the output for things that should not be present. Add as many objects to the scene"
                    "as needed, multiple copies of the same object are cheap and easy to do as long as the name is the"
                    "same. Include a camera object with position and rotation_euler_angles_degrees to set the camera"
                    "position and angle so that it captures the objects in the scene. Include the floor as a large flat"
                    "plane object which other objects will be on top of if it makes sense for the type of scene, e.g. a"
                    "space scene may not need a floor. Keep the size of objects close to a constant ratio between the"
                    "x, y, and z dimensions so that objects appear proportionate and realistic within the scene."
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


app = typer.Typer(add_completion=False)


@app.command()
def generate_scene(
    prompt: str,
    hunyuan_server_url: str | None = None,
    hunyuan_panorama_server_url: str | None = None,
    stitch_diffusion_server_url: str | None = None,
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
    import time

    scene_start_time = time.time()
    logger.debug(
        f"generate_scene command called with prompt: {prompt}, "
        f"output_dir: {output_dir}, model_batch_size: {model_batch_size}, "
        f"model_guidance_scale: {model_guidance_scale}"
    )

    # Generate scene layout
    layout_start = time.time()
    scene = generate_scene_object(prompt)
    layout_duration = time.time() - layout_start
    logger.debug(f"Scene object generated in {layout_duration:.2f}s, saving scene")

    save_scene_json(scene, output_dir)
    logger.debug("Scene saved, generating object models")

    # Generate object models
    models_start = time.time()
    generate_object_models(
        scene,
        output_dir,
        batch_size=model_batch_size,
        guidance_scale=model_guidance_scale,
        hunyuan_server_url=hunyuan_server_url,
    )
    models_duration = time.time() - models_start
    logger.debug(f"Object models generated in {models_duration:.2f}s, generating background")

    # Generate background
    background_start = time.time()
    generate_background_image(
        scene,
        output_dir,
        hunyuan_panorama_server_url=hunyuan_panorama_server_url,
        stitch_diffusion_server_url=stitch_diffusion_server_url,
    )
    background_duration = time.time() - background_start
    logger.debug(f"Background generated in {background_duration:.2f}s")

    # Total time
    total_duration = time.time() - scene_start_time
    logger.debug(
        f"Scene generation completed in {total_duration:.2f}s "
        f"(layout: {layout_duration:.2f}s, models: {models_duration:.2f}s, background: {background_duration:.2f}s)"
    )


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
def background(
    hunyuan_panorama_server_url: str | None = None,
    stitch_diffusion_server_url: str | None = None,
) -> None:
    """Generate the skybox image only (using Stable Diffusion)"""
    logger.debug("background command called")
    scene_file_path = "./Unity/AIML Research Project/Assets/scene.json"
    logger.debug(f"Reading scene from: {scene_file_path}")
    with open(scene_file_path) as f:
        scene_dict = json.load(f)
    s = Scene.model_validate(scene_dict)
    logger.debug("Scene loaded, generating background")
    generate_background_image(
        s,
        hunyuan_panorama_server_url=hunyuan_panorama_server_url,
        stitch_diffusion_server_url=stitch_diffusion_server_url,
    )
    logger.debug("Background generation completed")


@app.command(help="Generate scenes from a list of prompts in a file.")
def batch_generate(
    prompts_file: Path = Path("./prompts.txt"),
    output_dir: Path = Path("./generated_scenes_test"),
    hunyuan_server_url: str | None = None,
    hunyuan_panorama_server_url: str | None = None,
    stitch_diffusion_server_url: str | None = None,
    model_batch_size: int = 50,
    model_guidance_scale: float = 10.0,
) -> None:
    """
    Read prompts from a file (one per line) and generate scenes for each.

    Args:
        prompts_file (Path): Path to text file containing prompts (one per line).
        output_dir (Path): Base directory for generated scenes.
        hunyuan_server_url (str | None): URL of the Hunyuan server, if used.
        hunyuan_panorama_server_url (str | None): URL of the Hunyuan panorama server, if used.
        model_batch_size (int): Batch size for model generation.
        model_guidance_scale (float): Guidance scale for model generation.
    """
    from datetime import datetime

    logger.info(f"Reading prompts from: {prompts_file}")

    if not prompts_file.exists():
        logger.error(f"Prompts file not found: {prompts_file}")
        return

    # Read all prompts from file
    with open(prompts_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    logger.info(f"Found {len(prompts)} prompts to process")

    start_time = datetime.now()
    formatted_time = start_time.strftime("%Y-%m-%d_%H-%M-%S")

    for idx, prompt in enumerate(prompts, 1):
        logger.info(f"Processing prompt {idx}/{len(prompts)}: {prompt}")
        if prompt.startswith("#"):
            logger.info(f"Skipping comment prompt: {prompt}")
            continue

        safe_prompt = slugify(prompt)[:50]
        iteration_output_directory = output_dir / f"{formatted_time}-{idx:03d}-{safe_prompt}"
        iteration_output_directory.mkdir(parents=True, exist_ok=True)

        with open(iteration_output_directory / "prompt.txt", "w") as prompt_file:
            prompt_file.write(prompt)

        try:
            generate_scene(
                prompt,
                hunyuan_server_url=hunyuan_server_url,
                hunyuan_panorama_server_url=hunyuan_panorama_server_url,
                stitch_diffusion_server_url=stitch_diffusion_server_url,
                output_dir=iteration_output_directory,
                model_batch_size=model_batch_size,
                model_guidance_scale=model_guidance_scale,
            )
            logger.info(f"Successfully generated scene {idx}/{len(prompts)}")
        except Exception as e:
            logger.error(f"Failed to generate scene for prompt '{prompt}': {e}")
            continue

    end_time = datetime.now()
    duration = end_time - start_time
    logger.info("Batch generation complete")
    logger.info(f"Total prompts processed: {len(prompts)}")
    logger.info(f"Total time: {duration}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    app()
