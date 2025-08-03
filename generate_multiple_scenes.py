from pathlib import Path

import typer

from generate_scene import generate_scene

app = typer.Typer(add_completion=False)


@app.command()
def run(prompt: str, output_dir: Path = Path("./generated_scenes_test")) -> None:
    for i in range(3):
        iteration_output_directory = output_dir / str(i)
        iteration_output_directory.mkdir(parents=True, exist_ok=True)
        generate_scene(prompt, iteration_output_directory)


if __name__ == "__main__":
    app()
