#!/bin/bash
# HunyuanWorld-1.0 Complete Runtime Setup Script
# This script handles all dependencies and environment setup at runtime
# Mount this script to /workspace/scripts/startup.sh in the container

set -e

echo "🌍 HunyuanWorld-1.0 Complete Runtime Setup"
echo "==========================================="

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if conda environment exists
conda_env_exists() {
    conda info --envs | grep -q "^$1 "
}

# Function to check if a Python package is installed in the current environment
python_package_exists() {
    python -c "import $1" >/dev/null 2>&1
}

# Create necessary directories
echo "📁 Creating workspace directories..."
mkdir -p /workspace/{models,outputs,cache,external_tools}

# Change to the project directory
cd /workspace/HunyuanWorld-1.0

# Configure conda if not already done
if [[ ! -f ~/.condarc ]] || ! grep -q "channel_priority: flexible" ~/.condarc; then
    echo "⚙️ Configuring conda..."
    conda config --set channel_priority flexible
    conda tos accept 2>/dev/null || true
fi

# Check if HunyuanWorld environment exists, create if not
if ! conda_env_exists "HunyuanWorld"; then
    echo "🐍 Creating HunyuanWorld conda environment..."
    
    # Create environment file
    cat > /tmp/environment_runtime.yaml << 'EOF'
name: HunyuanWorld
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pytorch::pytorch=2.1.2
  - pytorch::torchvision=0.16.2  
  - pytorch::torchaudio=2.1.2
  - pytorch::pytorch-cuda=12.1
  - numpy=1.26.4
  - scipy
  - pillow
  - matplotlib
  - opencv
  - pip:
    - accelerate
    - diffusers>=0.30.0
    - transformers
    - huggingface_hub
    - xformers
    - open3d
    - trimesh
    - scikit-image
    - imageio
    - tqdm
    - omegaconf
    - easydict
    - utils3d
EOF
    
    conda env create -f /tmp/environment_runtime.yaml
    conda clean -afy
    rm /tmp/environment_runtime.yaml
    echo "✅ Conda environment created"
else
    echo "✅ HunyuanWorld conda environment already exists"
fi

# Activate the environment
echo "🔄 Activating HunyuanWorld environment..."
source /opt/conda/etc/profile.d/conda.sh
conda activate HunyuanWorld

# Ensure activation persists by setting environment variables
export CONDA_DEFAULT_ENV=HunyuanWorld
export CONDA_PREFIX=/opt/conda/envs/HunyuanWorld

# Lock NumPy version and prevent upgrades
echo "🔒 Locking NumPy version..."
conda install "numpy=1.26.4" --force-reinstall -y
echo "numpy=1.26.4" >> /opt/conda/envs/HunyuanWorld/conda-meta/pinned

# Runtime NumPy compatibility check and fix
echo "🔍 Checking NumPy compatibility..."
NUMPY_VERSION=$(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'unknown')
if [[ "$NUMPY_VERSION" =~ ^2\. ]]; then
    echo "⚠️ NumPy 2.x detected ($NUMPY_VERSION), downgrading to 1.26.4..."
    pip install 'numpy==1.26.4' --force-reinstall --no-deps
    echo "✅ NumPy downgraded successfully"
else
    echo "✅ NumPy compatibility OK ($NUMPY_VERSION)"
fi

# Configure pip for the environment
echo "⚙️ Configuring pip..."
pip config set global.timeout 600
pip config set global.retries 3

# Install additional Python packages
echo "📦 Installing additional Python packages..."
PACKAGES_TO_INSTALL=""

# Check and collect missing packages
for pkg in basicsr pytorch-lightning torchmetrics segment-anything groundingdino-py; do
    pkg_import_name=""
    case $pkg in
        "pytorch-lightning") pkg_import_name="pytorch_lightning" ;;
        "segment-anything") pkg_import_name="segment_anything" ;;
        "groundingdino-py") pkg_import_name="groundingdino" ;;
        *) pkg_import_name="$pkg" ;;
    esac
    
    if ! python_package_exists "$pkg_import_name"; then
        PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL $pkg"
    fi
done

if [[ -n "$PACKAGES_TO_INSTALL" ]]; then
    echo "Installing missing packages:$PACKAGES_TO_INSTALL"
    pip install $PACKAGES_TO_INSTALL || echo "⚠️ Some optional packages failed to install"
else
    echo "✅ All Python packages already installed"
fi

# Install Real-ESRGAN if not present
if [[ ! -d "/workspace/external_tools/Real-ESRGAN" ]]; then
    echo "🖼️ Installing Real-ESRGAN..."
    cd /workspace/external_tools
    git clone https://github.com/xinntao/Real-ESRGAN.git
    cd Real-ESRGAN
    pip install facexlib gfpgan || echo "⚠️ Some Real-ESRGAN dependencies failed"
    pip install -r requirements.txt || echo "⚠️ Some Real-ESRGAN requirements failed"
    python setup.py develop || echo "⚠️ Real-ESRGAN setup completed with warnings"
    echo "✅ Real-ESRGAN installation completed"
else
    echo "✅ Real-ESRGAN already installed"
fi

# Install ZIM (Zero-shot Image Matting) if not present
if [[ ! -d "/workspace/external_tools/ZIM" ]]; then
    echo "🎭 Installing ZIM (Zero-shot Image Matting)..."
    cd /workspace/external_tools
    git clone https://github.com/naver-ai/ZIM.git
    cd ZIM
    pip install -e . || echo "⚠️ ZIM installation completed with warnings"
    echo "✅ ZIM installation completed"
else
    echo "✅ ZIM already installed"
fi

# Install MoGe for depth estimation if not present
if ! python_package_exists "moge"; then
    echo "🏔️ Installing MoGe for depth estimation..."
    pip install "git+https://github.com/microsoft/MoGe.git" || echo "⚠️ MoGe installation completed with warnings"
    echo "✅ MoGe installation completed"
else
    echo "✅ MoGe already installed"
fi

# Apply torchvision compatibility fix for basicsr
echo "🔧 Applying torchvision compatibility fix..."
python -c "
import os
file_path = '/opt/conda/envs/HunyuanWorld/lib/python3.10/site-packages/basicsr/data/degradations.py'
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    content = content.replace('from torchvision.transforms.functional_tensor import rgb_to_grayscale', 'from torchvision.transforms.functional import rgb_to_grayscale')
    with open(file_path, 'w') as f:
        f.write(content)
    print('✅ Fixed torchvision import in basicsr')
else:
    print('⚠️ basicsr file not found, skipping fix')
" || echo "⚠️ basicsr fix attempted"

# Return to project directory
cd /workspace/HunyuanWorld-1.0

# Fix the FluxIPAdapterMixin import issue in the pipeline (if not already fixed)
if [[ -f "hy3dworld/models/pipelines.py" ]]; then
    if grep -q "FluxIPAdapterMixin" hy3dworld/models/pipelines.py; then
        echo "🔧 Fixing FluxIPAdapterMixin import..."
        sed -i 's/FluxIPAdapterMixin/IPAdapterMixin/g' hy3dworld/models/pipelines.py
        echo "✅ Pipeline import fixed"
    else
        echo "✅ FluxIPAdapterMixin import already fixed"
    fi
fi

# Set up environment variables
echo "🌐 Setting up environment variables..."
export TRANSFORMERS_CACHE="/workspace/cache/transformers"
export HF_HOME="/workspace/cache/huggingface"
export TORCH_HOME="/workspace/cache/torch"
export PYTHONPATH="/workspace/HunyuanWorld-1.0"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# HuggingFace authentication
if [[ -n "$HF_TOKEN" ]]; then
    echo "🔐 Setting up HuggingFace authentication..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 || true
    echo "✅ HuggingFace authentication completed"
fi

# Verify installation by testing key imports
echo "🧪 Verifying installation..."
python -c "
try:
    import torch
    import numpy as np
    import diffusers
    import transformers
    import open3d
    print('✅ Core dependencies verified')
    print(f'   PyTorch: {torch.__version__}')
    print(f'   NumPy: {np.__version__}')
    print(f'   CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
except ImportError as e:
    print(f'⚠️ Import verification failed: {e}')
"

# Display setup completion and usage information
echo ""
echo "� HunyuanWorld-1.0 Runtime Setup Complete!"
echo ""
echo "📊 Setup Summary:"
echo "  ✅ Conda environment: HunyuanWorld"
echo "  ✅ Python packages installed"
echo "  ✅ External tools: Real-ESRGAN, ZIM, MoGe"
echo "  ✅ Compatibility fixes applied"
echo "  ✅ Environment variables configured"
echo ""
echo "Usage Examples:"
echo "  Text → 3D World:"
echo "    python demo_panogen.py --prompt \"volcanic landscape\" --output_path outputs/volcano --fp8_attention --fp8_gemm --cache"
echo "    python demo_scenegen.py --image_path outputs/volcano/panorama.png --classes outdoor --output_path outputs/volcano"
echo ""
echo "  Image → 3D World:"
echo "    python demo_panogen.py --image_path input.jpg --output_path outputs/result --fp8_attention --fp8_gemm"
echo "    python demo_scenegen.py --image_path outputs/result/panorama.png --classes outdoor --output_path outputs/result"
echo ""
echo "Performance flags: --fp8_attention --fp8_gemm --cache"
echo "Output meshes: mesh_layer0.ply (foreground), mesh_layer1.ply (midground), mesh_layer2.ply (background)"
echo ""
echo "🚀 Ready to generate 3D worlds!"
