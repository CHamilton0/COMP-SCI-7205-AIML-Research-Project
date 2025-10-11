import base64
import json
import logging
import re
import subprocess
import tempfile
import warnings
from pathlib import Path

import requests
import typer
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Hide future warnings from shap_e
warnings.filterwarnings("ignore", category=FutureWarning)

app = typer.Typer(add_completion=False)


class Vector3(BaseModel):
    x: float
    y: float
    z: float


class SceneObject(BaseModel):
    name: str
    position: Vector3
    size: Vector3
    rotation_euler_angles_degrees: Vector3


class Scene(BaseModel):
    stable_diffusion_scene_skybox_prompt: str
    objects: list[SceneObject]
    camera: SceneObject


def generate_scene_object(prompt: str) -> Scene:
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
                    "Imagine a scene based on the prompt with some objects in the scene."
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


def save_scene(scene: Scene, output_dir: Path = Path("./Unity/AIML Research Project/Assets")) -> None:
    logger.debug(f"save_scene called with output_dir: {output_dir}")
    scene_json = scene.model_dump_json(indent=4)
    scene_file_path = output_dir / "scene.json"
    logger.debug(f"Writing scene to file: {scene_file_path}")
    with open(scene_file_path, "w") as scene_file:
        scene_file.write(scene_json)
    logger.debug("Scene successfully saved to file")


def generate_background(scene: Scene, output_dir: Path = Path("./Unity/AIML Research Project/Assets")) -> None:
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


# Generate a safe filename from the prompt
def slugify(text: str) -> str:
    logger.debug(f"slugify called with text: {text}")
    result = re.sub(r"[^a-zA-Z0-9\-]+", "-", text.strip().lower()).strip("-")
    logger.debug(f"slugify result: {result}")
    return result


def generate_shap_e_model(
    object_name: str,
    output_file: Path,
    model,
    diffusion,
    xm,
    device,
    batch_size: int = 1,
    guidance_scale: float = 15.0,
) -> None:
    from shap_e.diffusion.sample import sample_latents
    from shap_e.util.notebooks import decode_latent_mesh

    logger.debug(f"Sampling latents for object: {object_name}")
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[object_name] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    logger.debug("Latent sampling completed")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        ply_file = tmp_path / "mesh.ply"

        print(f"Saving intermediate PLY to: {ply_file}")
        logger.debug(f"Decoding latent mesh to PLY: {ply_file}")
        t = decode_latent_mesh(xm, latents[0]).tri_mesh()
        with open(ply_file, "wb") as f:
            t.write_ply(f)
        logger.debug("PLY file written successfully")

        # Call Blender to convert PLY → GLB
        blender_script = Path("./Blender/export.py").resolve()
        command = [
            "blender",
            "-b",
            "-P",
            str(blender_script),
            "--",
            str(ply_file),
            str(output_file),
        ]
        print("Running Blender export script...")
        logger.debug(f"Running Blender command: {' '.join(command)}")
        subprocess.run(command, check=True)
        logger.debug("Blender export completed successfully")


def generate_hunyuan_model(
    hunyuan_world_server_url: str,
    object_name: str,
    output_file: Path,
) -> None:
    object_request = requests.post(
        f"{hunyuan_world_server_url}/generate",
        json={"text": object_name, "texture": True},
        headers={"Content-Type": "application/json"},
    )
    with open(output_file, "wb") as f:
        f.write(object_request.content)


def generate_object_models(
    scene: Scene,
    output_dir: Path = Path("./Unity/AIML Research Project/Assets"),
    batch_size: int = 1,  # the size of the models, higher values take longer to generate
    guidance_scale: float = 15.0,  # the scale of the guidance, higher values make the model look more like the prompt
    use_shap_e: bool = False,
) -> None:
    logger.debug(
        f"generate_object_models called with output_dir: {output_dir}, "
        f"batch_size: {batch_size}, guidance_scale: {guidance_scale}, use_shap_e: {use_shap_e}"
    )
    glb_dir = output_dir / "GLB"
    glb_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created GLB directory: {glb_dir}")

    if use_shap_e:
        import torch
        from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
        from shap_e.models.download import load_config, load_model

        print("Loading models...")
        logger.debug("Loading Shap-E models")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Using device: {device}")
        xm = load_model("transmitter", device=device)
        model = load_model("text300M", device=device)
        diffusion = diffusion_from_config(load_config("diffusion"))
        print("Done loading models...")
        logger.debug("Successfully loaded all Shap-E models")

    object_names = set(object.name for object in scene.objects)
    logger.debug(f"Unique object names to generate: {object_names}")

    print("Generating models for scene")
    for object_name in object_names:
        logger.debug(f"Starting generation for object: {object_name}")
        base_name = slugify(object_name)
        output_file = glb_dir / f"{base_name}.glb"

        print(f"Generating model for prompt: '{object_name}'")
        if use_shap_e:
            generate_shap_e_model(
                object_name=object_name,
                output_file=output_file,
                model=model,
                diffusion=diffusion,
                xm=xm,
                device=device,
                batch_size=batch_size,
                guidance_scale=guidance_scale,
            )
        else:
            generate_hunyuan_model(
                hunyuan_world_server_url="http://10.12.65.80:8080", object_name=object_name, output_file=output_file
            )

        logger.debug(f"Successfully generated model file: {output_file}")

    logger.debug("All object models generated successfully")


class SceneImageAnalysisResult(BaseModel):
    objects_in_first_image: list[str]
    objects_in_second_image: list[str]
    first_image_better_scene: bool


def analyse_scene_images(
    image_path1: Path,
    image_path2: Path,
    scene_prompt: str,
) -> SceneImageAnalysisResult:
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
    output_dir: Path = Path("./Unity/AIML Research Project/Assets"),
    model_batch_size: int = 1,
    model_guidance_scale: float = 15.0,
    use_shap_e: bool = False,
) -> None:
    logger.debug(
        f"generate_scene command called with prompt: {prompt}, "
        f"output_dir: {output_dir}, model_batch_size: {model_batch_size}, "
        f"model_guidance_scale: {model_guidance_scale}"
    )
    scene = generate_scene_object(prompt)
    logger.debug("Scene object generated, saving scene")
    save_scene(scene, output_dir)
    logger.debug("Scene saved, generating object models")
    generate_object_models(scene, output_dir, model_batch_size, model_guidance_scale, use_shap_e)
    logger.debug("Object models generated, generating background")
    generate_background(scene, output_dir)
    print(f"Scene generation complete. Output in {output_dir}")
    logger.debug("Scene generation completed successfully")


@app.command()
def generate_scene_layout(prompt: str) -> None:
    logger.debug(f"generate_scene_layout called with prompt: {prompt}")
    scene = generate_scene_object(prompt)
    scene_json = scene.model_dump_json(indent=4)
    print(scene_json)
    logger.debug("Scene layout generated and printed")


@app.command(help="Generate GLB models for each object from the existing scene.json.")
def models() -> None:
    """Read scene.json and generate GLB models for unique object names."""
    logger.debug("models command called")
    scene_file_path = "./Unity/AIML Research Project/Assets/scene.json"
    logger.debug(f"Reading scene from: {scene_file_path}")
    with open(scene_file_path) as f:
        scene_dict = json.load(f)
    s = Scene.model_validate(scene_dict)
    logger.debug(f"Scene loaded with {len(s.objects)} objects")
    generate_object_models(s)
    logger.debug("Model generation completed")


@app.command(help="Generate a background image using the skybox prompt from scene.json.")
def background() -> None:
    """Generate the skybox image only (using Stable Diffusion)."""
    logger.debug("background command called")
    scene_file_path = "./Unity/AIML Research Project/Assets/scene.json"
    logger.debug(f"Reading scene from: {scene_file_path}")
    with open(scene_file_path) as f:
        scene_dict = json.load(f)
    s = Scene.model_validate(scene_dict)
    logger.debug("Scene loaded, generating background")
    generate_background(s)
    logger.debug("Background generation completed")


if __name__ == "__main__":
    app()
