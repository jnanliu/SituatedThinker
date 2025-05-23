set -exo pipefail

export TOKENIZERS_PARALLELISM=False

python -m data.build_grpo_data