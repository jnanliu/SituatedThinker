set -exo pipefail

MODEL_NAME=$1
TP=$2

vllm serve $MODEL_NAME \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size $TP \
    --gpu-memory-utilizatio 0.9