import base64
import logging
import os
from pathlib import Path

import typer

from scene_classes import SceneImageAnalysisResult

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


app = typer.Typer(add_completion=False)


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


if __name__ == "__main__":
    app()
