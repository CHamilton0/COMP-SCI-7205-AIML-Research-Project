import re
import subprocess
import tempfile
from pathlib import Path

import typer
from pydantic import BaseModel

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


def generate_scene_object(prompt: str) -> Scene:
    from openai import OpenAI

    openai_client = OpenAI()
    response = openai_client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {
                "role": "system",
                "content": "Imagine a scene based on the prompt with some objects in the scene."
                + "Extract the scene information. Ensure the prompt for the scene skybox could generate a skybox image"
                + "that could be used on a spherical shader. Add as many objects to the scene as needed, multiple"
                + "copies of the same object are cheap and easy to do.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        text_format=Scene,
    )

    scene = response.output_parsed
    return scene


def save_scene(scene: Scene) -> None:
    scene_json = scene.model_dump_json(indent=4)
    with open("./Unity/AIML Research Project/Assets/scene.json", "w") as scene_file:
        scene_file.write(scene_json)


def generate_background(scene: Scene) -> None:
    import torch
    from diffusers import StableDiffusionPipeline

    # Replace the model version with your required version if needed
    pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16)

    # Running the inference on GPU with cuda enabled
    pipeline = pipeline.to("cuda")

    prompt = scene.stable_diffusion_scene_skybox_prompt

    images = pipeline(prompt=prompt).images
    image = images[0]
    image.save("./Unity/AIML Research Project/Assets/background.png")


batch_size = 1  # this is the size of the models, higher values take longer to generate.
guidance_scale = 15.0  # this is the scale of the guidance, higher values make the model look more like the prompt.

output_dir = Path("./Unity/AIML Research Project/Assets/GLB")
output_dir.mkdir(parents=True, exist_ok=True)


# Generate a safe filename from the prompt
def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\-]+", "-", text.strip().lower()).strip("-")


def generate_object_models(scene: Scene) -> None:
    import torch
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.diffusion.sample import sample_latents
    from shap_e.models.download import load_config, load_model
    from shap_e.util.notebooks import decode_latent_mesh

    print("Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))
    print("Done loading models...")

    object_names = set(object.name for object in scene.objects)

    for object_name in object_names:
        base_name = slugify(object_name)
        output_file = output_dir / f"{base_name}.glb"

        print(f"[+] Generating model for prompt: '{object_name}'")
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

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ply_file = tmp_path / "mesh.ply"

            print(f"[+] Saving intermediate PLY to: {ply_file}")
            t = decode_latent_mesh(xm, latents[0]).tri_mesh()
            with open(ply_file, "wb") as f:
                t.write_ply(f)

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
            print("[+] Running Blender export script...")
            subprocess.run(command, check=True)

        print(f"[✓] Exported: {output_file}")


@app.command()
def generate_scene(prompt: str):
    scene = generate_scene_object(prompt)
    save_scene(scene)
    generate_object_models(scene)
    generate_background(scene)
    print("Scene generation complete.")


@app.command()
def generate_scene_layout(prompt: str):
    scene = generate_scene_object(prompt)
    print(scene.model_dump_json(indent=4))


@app.command(help="Generate GLB models for each object from the existing scene.json.")
def models():
    """Read scene.json and generate GLB models for unique object names."""
    import json

    with open("./Unity/AIML Research Project/Assets/scene.json") as f:
        scene_dict = json.load(f)
    s = Scene.model_validate(scene_dict)
    generate_object_models(s)


@app.command(help="Generate a background image using the skybox prompt from scene.json.")
def background():
    """Generate the skybox image only (using Stable Diffusion)."""
    import json

    with open("./Unity/AIML Research Project/Assets/scene.json") as f:
        scene_dict = json.load(f)
    s = Scene.model_validate(scene_dict)
    generate_background(s)


if __name__ == "__main__":
    app()
