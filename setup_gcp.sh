#!/bin/bash
# GCP Setup Script for KV Cache Benchmark
# Run this once after SSH-ing into your GCP instance.
# Usage: bash setup_gcp.sh
set -e

echo "=== KV Benchmark GCP Setup ==="

# Update and install essentials
sudo apt-get update -q
sudo apt-get install -y git tmux htop nvtop

# Verify GPU
nvidia-smi
echo "GPU check passed"

# Navigate to home
cd /home/$USER

# Install Python dependencies
pip install --upgrade pip
pip install \
    "torch>=2.1.0" \
    "transformers>=4.40.0" \
    "datasets>=2.18.0" \
    "accelerate>=0.27.0" \
    "pyyaml>=6.0" \
    "numpy>=1.24.0" \
    "matplotlib>=3.7.0" \
    "pandas>=2.0.0" \
    "rouge-score>=0.1.2" \
    "tqdm>=4.65.0" \
    "tabulate>=0.9.0"

# HuggingFace login for LLaMA-2 access
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Set it with: export HF_TOKEN=your_token"
    echo "Then run: huggingface-cli login --token \$HF_TOKEN"
else
    echo "Logging into HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN"

    # Pre-download model weights to avoid timeout during benchmark
    echo "Pre-downloading LLaMA-2-7B weights (this may take a few minutes)..."
    python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
print('Downloading model weights...')
AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', torch_dtype='float16')
print('Model download complete')
"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Recommended workflow:"
echo "  1. Start a tmux session:  tmux new -s benchmark"
echo "  2. Run smoke test:        python run_benchmark.py --dry_run"
echo "  3. Run full benchmark:    python run_benchmark.py"
echo "  4. Detach tmux:           Ctrl+B then D"
echo "  5. Reattach later:        tmux attach -t benchmark"
echo ""
echo "Monitor GPU in a second window:"
echo "  tmux new-window && watch -n 2 nvidia-smi"
