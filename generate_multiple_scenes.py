from datetime import datetime
from pathlib import Path

import typer

from generate_scene import generate_scene

app = typer.Typer(add_completion=False)


@app.command()
def run(
    prompt: str,
    output_dir: Path = Path("./generated_scenes_test"),
) -> None:
    start_time = datetime.now()
    formatted_time = start_time.strftime("%Y-%m-%d_%H-%M-%S")

    for model_batch_size in [1, 2, 5, 10]:
        for model_guidance_scale in [1.0, 10.0, 100.0]:
            iteration_output_directory = (
                output_dir / f"{formatted_time}-batchsize-{model_batch_size}-guidancescale-{model_guidance_scale}"
            )
            iteration_output_directory.mkdir(parents=True, exist_ok=True)

            with open(iteration_output_directory / "prompt.txt", "w") as prompt_file:
                prompt_file.write(prompt)

            generate_scene(prompt, iteration_output_directory)


if __name__ == "__main__":
    app()
