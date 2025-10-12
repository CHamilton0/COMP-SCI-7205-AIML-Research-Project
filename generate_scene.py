import base64
import json
import logging
import warnings
from pathlib import Path

import typer

from scene_classes import Scene, SceneImageAnalysisResult
from model_generation import generate_object_models

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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


def analyse_scene_images(
    image_path1: Path,
    image_path2: Path,
    scene_prompt: str,
) -> SceneImageAnalysisResult:
    """
    Analyze two scene images and determine which better represents the scene described by the prompt.
    Args:
        image_path1 (Path): The file path to the first image.
        image_path2 (Path): The file path to the second image.
        scene_prompt (str): The text prompt describing the scene.
    Returns:
        SceneImageAnalysisResult: The analysis result indicating which image is better.
    """
    logger.debug(f"get_objects_in_scene called with image_paths: {image_path1} and {image_path2}")
    from openai import OpenAI

    def encode_image(image_path: Path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    openai_client = OpenAI()
    logger.debug("Generating object list with LLM")
    response = openai_client.responses.parse(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Which of the two following images better represent the scene described by the"
                            f"following prompt? {scene_prompt}"
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{encode_image(image_path1)}",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{encode_image(image_path2)}",
                    },
                ],
            },
        ],
        text_format=SceneImageAnalysisResult,
    )

    result: SceneImageAnalysisResult = response.output_parsed

    return result


app = typer.Typer(add_completion=False)


@app.command()
def get_best_scene(
    scene_prompt: str,
    generated_scenes_directory: Path = Path("./Unity/AIML Research Project/Assets/generated-scenes"),
) -> None:
    """
    Compare generated scene images and select the best one based on the scene prompt.
    Args:
        scene_prompt (str): The text prompt describing the scene.
        generated_scenes_directory (Path): The directory containing generated scene images.
    """

    logger.debug(f"get_best_scene command called with generated_scenes_directory: {generated_scenes_directory}")
    scene_directories = sorted(
        [d for d in generated_scenes_directory.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )[:2]

    if not scene_directories or len(scene_directories) < 2:
        logger.error("Not enough scene directories found.")
        return

    # Create pairs of images to compare until only one best scene remains
    while len(scene_directories) > 1:
        dir1 = scene_directories.pop()
        dir2 = scene_directories.pop()

        image1 = dir1 / "screenshot.png"
        image2 = dir2 / "screenshot.png"

        if not image1.exists() or not image2.exists():
            logger.warning(f"One of the images does not exist: {image1}, {image2}")
            continue

        logger.debug(f"Comparing images: {image1} and {image2}")
        analysis_result = analyse_scene_images(image1, image2, scene_prompt)

        if analysis_result.first_image_better_scene:
            scene_directories.append(dir1)  # Keep the better scene for further comparison
            logger.info(f"Selected better scene: {dir1}")
        else:
            scene_directories.append(dir2)  # Keep the better scene for further comparison
            logger.info(f"Selected better scene: {dir2}")

    if scene_directories:
        logger.info(f"Best scene found: {scene_directories[0]}")


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
