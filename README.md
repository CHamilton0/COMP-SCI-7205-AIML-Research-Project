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
uv run python generate_scene.py generate-scene "a space scene"  <options>
```

Run:

```bash
uv run python generate_scene.py generate-scene --help
```

for usage on the available options.

## Batch Scene Generation

Add scene prompts to the `prompts.txt` file in this directory

```bash
uv run python generate_scene.py batch-generate <options>
```

The scene generation commands have options to pass in a URL for the Hunyuan3D model, the HunyuanWorld panorama model, or
the Diffusion360 model. These models have been containerised so that APIs for each can run on available hardware to
avoid memory issues and improve performance.

Run:

```bash
uv run python generate_scene.py batch-generate --help
```

for usage on the available options.

## Running modules

### Hunyuan3D

The Hunyuan3D project has been included as a submodule in this repository. A Dockerfile has been written for this
repository to containerise the API server functionality, and allow easily running it on any machine with Docker.

To run the module, run:


```bash
cd modules/Hunyuan3D-2/
docker run -d --name hunyuan3d-server --gpus all -p 8080:8080 -v ./cache/hunyuan3d_cache:/root/.cache -v ./cache/u2net_cache:/root/.u2net hunyuan3d-2:latest --enable_tex
```

This will run the Hunyuan server on port 8080, and the flag: `--hunyuan-server-url` can be passed to the
`generate-scene` or `batch-generate` commands in the form of `http://<host>:8080` to tell the scene composer to use this
server to generate 3D objects.

## HunyuanWorld

The HunyuanWorld project has been included as a submodule in this repository. A Dockerfile has been written for this
repository to containerise the API server functionality, and allow easily running it on any machine with Docker.

To run the module using Docker Compose, run:


```bash
cd modules/HunyuanWorld-1.0/
docker compose -f docker-compose.runtime.yml up -d
```

This will run the HunyuanWorld server but at this stage it has not been integrated into the scene composer as part of
full scene generation. Inside of the container, it is possible to run commands like:

```bash
python demo_panogen.py --prompt "volcanic landscape" --output_path outputs/volcano --fp8_attention --fp8_gemm --cache
python demo_scenegen.py --image_path outputs/volcano/panorama.png --classes outdoor --output_path outputs/volcano
```

And then generated scenes can be copied out using `docker cp` commands.

## HunyuanWorld Panorama API

To make it easy to be able to generate HunyuanWorld panoramas from other services, a REST API was set up.

To run this using Docker Compose, run:


```bash
cd modules/HunyuanWorldAPI/
docker compose -f docker-compose.api.yml up -d
```

This will run the HunyuanWorld panorama API server on port 8000, and the flag: `--hunyuan-panorama-server-url` can be
passed to the `generate-scene` or `batch-generate` commands in the form of `http://<host>:8000` to tell the scene
composer to use this server to generate panoramic backgrounds.

## Diffusion360

The Diffusion360 project has been included as a submodule in this repository. A Dockerfile has been written for this
repository to containerise the API server functionality, and allow easily running it on any machine with Docker.

To run the module using Docker Compose, run:


```bash
cd modules.SD-T2I-360PanoImage/
docker compose up -d
```

This will run the HunyuanWorld panorama API server on port 8000, and the flag: `--diffusion360-server-url` can be
passed to the `generate-scene` or `batch-generate` commands in the form of `http://<host>:8000` to tell the scene
composer to use this server to generate panoramic backgrounds.
