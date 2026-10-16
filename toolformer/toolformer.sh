#!/bin/bash
#SBATCH --job-name=toolformer_test
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20            # 24 CPU cores
#SBATCH --mem=64G                    # 64GB RAM
#SBATCH --time=01:30:00              # 
#SBATCH --output=toolformer_eval_%j.log

set -euo pipefail

# 1. Environment Setup

module load cuda
module use ~/.local/easybuild/modules/all
module load Anaconda3

source "/home/ctheod03/.local/easybuild/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh"
conda activate "/home/ctheod03/.local/easybuild/software/Anaconda3/2025.06-1/envs/env"

export PATH="/home/ctheod03/.local/easybuild/software/Anaconda3/2025.06-1/envs/env/bin:$PATH"
hash -r


echo "Python being used:"
which python
python --version

echo "OpenAI check:"
python -c "import sys; print(sys.executable)"
python -c "import openai; print(openai.__version__)"

# 2. API Key Deployment
export WOLFRAM_ALPHA_APPID="3P8LV54EG6"

# 3. Forcefully override all cluster-wide read-only Hugging Face & vLLM targets
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/transformers"
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"
export VLLM_ATTENTION_BACKEND=TRITON_ATTN


# 4. Spin up the background vLLM engine with an explicit local download path flag
echo "==============================================="
echo "Starting background vLLM engine with GPT-J-6B..."
echo "==============================================="
vllm serve EleutherAI/gpt-j-6b \
    --host 127.0.0.1 \
    --port 5000 \
    --dtype float16 \
    --max-model-len 2048 \
    --tensor-parallel-size 1 \
    --download-dir "$HOME/.cache/huggingface/hub" &

VLLM_PID=$!

# 5. Wait for the engine to download weights and boot
echo "Waiting for vLLM to fetch weights from Hugging Face and mount to GPU..."
until curl -s http://127.0.0.1:5000/v1/models > /dev/null; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM crashed during startup - check the log above"; exit 1
    fi
    sleep 10
done

echo "vLLM server is officially online and healthy!"

# # 6. TEST 1: Run Single-Batch Latency Benchmark
# echo "==============================================="
# echo "TEST 1: Running Single-Batch Latency Benchmark"
# echo "==============================================="

# # python math_toolformer.py

# echo "Finished Test 1. Short pause..."
# sleep 5

# 7. TEST 2: Run Multi-Batch Throughput Benchmark
echo "==============================================="
echo "TEST 2: Running Multi-Batch Throughput Benchmark"
echo "==============================================="

python batch_toolformer.py --output ./nomaxtokens_triton_1t4.json --timeout 2800

# 8. Cluster Cleanup
echo "==============================================="
echo "Testing complete! Shutting down background processes..."
echo "==============================================="
kill $VLLM_PID
