import logging
import re
import subprocess
import tempfile
from enum import Enum
from pathlib import Path

import requests

from scene_classes import Scene

logger = logging.getLogger(__name__)


class ObjectModel(str, Enum):
    """
    Indicates the model to use for object generation.
    """

    SHAP_E = "shap_e"
    HUNYUAN = "hunyuan"


def slugify(text: str) -> str:
    """
    Converts a string into a slugified version for filenames.
    Args:
        text (str): The input text to slugify.
    Returns:
        str: The slugified text.
    """

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
    batch_size: int = 1,
    guidance_scale: float = 15.0,
) -> None:
    """
    Generate a 3D model using Shap-E. Places the generated GLB file at output_file.
    Args:
        object_name (str): The text prompt for the object to generate
        output_file (Path): The path to save the generated GLB file
    Returns:
        None
    """

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

        logger.debug(f"Decoding latent mesh to PLY: {ply_file}")
        t = decode_latent_mesh(xm, latents[0]).tri_mesh()
        with open(ply_file, "wb") as f:
            t.write_ply(f)
        logger.debug("PLY file written successfully")

        # Call Blender to convert PLY to GLB
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
        logger.debug(f"Running Blender command: {' '.join(command)}")
        subprocess.run(command, check=True)
        logger.debug("Blender export completed successfully")


def generate_hunyuan_model(
    hunyuan_server_url: str,
    object_name: str,
    output_file: Path,
) -> None:
    """
    Generate a 3D model using the Hunyuan World server. Places the generated GLB file at output_file.

    Args:
        hunyuan_server_url (str): The base URL of the Hunyuan World server
        object_name (str): The text prompt for the object to generate
        output_file (Path): The path to save the generated GLB file
    Returns:
        None
    """

    object_request = requests.post(
        f"{hunyuan_server_url}/generate",
        json={"text": object_name, "texture": True},
        headers={"Content-Type": "application/json"},
    )
    with open(output_file, "wb") as f:
        f.write(object_request.content)


def generate_object_models(
    scene: Scene,
    output_dir: Path = Path("./Unity/AIML Research Project/Assets"),
    model: ObjectModel = ObjectModel.SHAP_E,
    server_url: str | None = None,
    batch_size: int = 1,  # the size of the models, higher values take longer to generate
    guidance_scale: float = 15.0,  # the scale of the guidance, higher values make the model look more like the prompt
) -> None:
    """
    Generate 3D models for each unique object in the scene using either Shap-E or Hunyuan World into the output
    directory under GLB
    Args:
        scene (Scene): The scene containing objects to generate
        output_dir (Path): The directory to save the generated GLB files
        model (ObjectModel): The model to use for object generation
        server_url (str | None): The URL of the server for remote models
        batch_size (int): The size of the models, higher values take longer to generate
        guidance_scale (float): The scale of the guidance, higher values make the model look more like the prompt
    Returns:
        None
    """

    logger.debug(
        f"generate_object_models called with output_dir: {output_dir}, "
        f"batch_size: {batch_size}, guidance_scale: {guidance_scale}, model: {model}, server_url: {server_url}"
    )
    glb_dir = output_dir / "GLB"
    glb_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created GLB directory: {glb_dir}")

    if model == ObjectModel.SHAP_E:
        # Set up the Shap-E model
        import torch
        from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
        from shap_e.models.download import load_config, load_model

        logger.debug("Loading Shap-E models")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Using device: {device}")
        xm = load_model("transmitter", device=device)
        shap_e_model = load_model("text300M", device=device)
        diffusion = diffusion_from_config(load_config("diffusion"))
        logger.debug("Successfully loaded all Shap-E models")
    elif model == ObjectModel.HUNYUAN:
        # Ensure Hunyuan3D server URL is provided if the model is selected
        if server_url is None:
            raise ValueError("server_url is required for HUNYUAN model")

    object_names = set(object.name for object in scene.objects)
    logger.debug(f"Unique object names to generate: {object_names}")

    logger.debug("Generating models for scene")
    for object_name in object_names:
        logger.debug(f"Starting generation for object: {object_name}")
        base_name = slugify(object_name)
        output_file = glb_dir / f"{base_name}.glb"

        if model == ObjectModel.SHAP_E:
            generate_shap_e_model(
                object_name=object_name,
                output_file=output_file,
                model=shap_e_model,
                diffusion=diffusion,
                xm=xm,
                batch_size=batch_size,
                guidance_scale=guidance_scale,
            )
        elif model == ObjectModel.HUNYUAN:
            generate_hunyuan_model(hunyuan_server_url=server_url, object_name=object_name, output_file=output_file)
        else:
            raise ValueError(f"Unknown object model: {model}")

        logger.debug(f"Successfully generated model file: {output_file}")

    logger.debug("All object models generated successfully")
