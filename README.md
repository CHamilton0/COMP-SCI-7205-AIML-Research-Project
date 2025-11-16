# COMP-SCI-7205-AIML-Research-Project

## Requirements

This repository includes the use of models that are expecting to have access to a CUDA-compatible GPU.

## Setup

### Install Blender

```bash
cd /opt
sudo wget https://download.blender.org/release/Blender3.6/blender-3.6.17-linux-x64.tar.xz
sudo tar -xf blender-3.6.17-linux-x64.tar.xz
sudo ln -s /opt/blender-3.6.17-linux-x64/blender /usr/local/bin/blender
sudo rm blender-3.6.17-linux-x64.tar.xz
sudo apt install -y libsm6 libx11-6 libxi6 libxext6 libxfixes3 libxxf86vm1 libxrender1 libxrandr2 libxinerama1 libxcursor1
```

## Run Scene Generation

Use the LOG_LEVEL environment variable to change the log level. E.g.

```bash
export LOG_LEVEL=debug
```

```bash
uv run python generate_scene.py generate-scene "a space scene" 
```

## Batch Scene Generation

Add scene prompts to the `prompts.txt` file in this directory

```bash
uv run python generate_scene.py batch-generate
```

The scene generation commands have options to pass in a URL for the Hunyuan3D model, the HunyuanWorld panorama model, or
the Diffusion360 model. These models have been containerised so that APIs for each can run on available hardware to
avoid memory issues and improve performance.
