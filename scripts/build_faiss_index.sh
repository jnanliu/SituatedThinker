set -exo pipefail

export TOKENIZERS_PARALLELISM=False

python -m data.build_faiss_index --dataset wikipedia18