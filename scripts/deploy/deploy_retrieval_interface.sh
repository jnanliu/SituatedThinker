set -exo pipefail

export PYTHONPATH=$PYTHONPATH:$(pwd)

python -m interfaces.retrieval.api