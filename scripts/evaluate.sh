set -exo pipefail

export TOKENIZERS_PARALLELISM=False

python -m evaluation.$1 \
    --checkpoint_dir $2 \
    --output_path $3 \
    --tensor_parallel_size $4 \
    --n $5 \
    --temperature $6 
    