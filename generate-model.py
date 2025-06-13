import re
import subprocess
import tempfile
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def convert(
    prompt: str = typer.Option(help="Prompt to generate 3D object"),
    output_dir: Path = typer.Option(
        help="Directory to output to.",
        default=Path("./Unity/AIML Research Project/Assets/Models"),
    ),
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a safe filename from the prompt
    def slugify(text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9\-]+", "-", text.strip().lower()).strip("-")

    base_name = slugify(prompt)
    output_file = output_dir / f"{base_name}.glb"

    import torch
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.diffusion.sample import sample_latents
    from shap_e.models.download import load_config, load_model
    from shap_e.util.notebooks import decode_latent_mesh

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    batch_size = (
        1  # this is the size of the models, higher values take longer to generate.
    )
    guidance_scale = 15.0  # this is the scale of the guidance, higher values make the model look more like the prompt.

    typer.echo(f"[+] Generating model for prompt: '{prompt}'")
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
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

        typer.echo(f"[+] Saving intermediate PLY to: {ply_file}")
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
        typer.echo(f"[+] Running Blender export script...")
        subprocess.run(command, check=True)

    typer.echo(f"[✓] Exported: {output_file}")


if __name__ == "__main__":
    app()
