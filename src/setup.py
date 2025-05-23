# Modified from veRL
#
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# setup.py is the fallback installation script when pyproject.toml does not work
from setuptools import setup, find_packages
import os

__version__ = "0.1.0"

install_requires = [
  "accelerate==1.4.0",
  "antlr4-python3-runtime==4.9.3",
  "codetiming==1.4.0",
  "datasets==3.3.2",
  "dill==0.3.8",
  "e2b==1.1.0",
  "e2b-code-interpreter==1.0.5",
  "faiss-gpu-cu12==1.10.0",
  "fastapi==0.115.11",
  "flashinfer-python==0.2.5",
  "FlagEmbedding==1.3.4",
  "httpx==0.28.1",
  "hydra-core==1.3.2",
  "liger-kernel==0.5.3",
  "latex2sympy2_extended==1.10.1",
  "llama-index==0.12.37",
  "math_verify==0.7.0"
  "ninja",
  "numpy==1.26.4",
  "pandas==2.2.3",
  "peft==0.14.0",
  "polars==1.29.0",
  "pyarrow==19.0.1",
  "pybind11==2.13.6",
  "pylatexenc==2.10",
  "ray==2.43.0",
  "sandbox-fusion==0.3.7",
  "sentence-transformers==3.4.1",
  "swanlab==0.5.3",
  "tabulate==0.9.0",
  "tensordict==0.6.2",
  "textworld==1.6.2",
  "torchdata==0.11.0",
  "transformers==4.51.3",
  "vllm==0.8.4",
  "wandb==0.19.7",
]

from pathlib import Path
this_directory = Path(__file__).parents[1]
long_description = (this_directory / "README.md").read_text()

setup(
    name="situatedthinker",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    url="https://github.com/jnanliu/SituatedThinker",
    license="Apache 2.0",
    author="Junnan Liu",
    author_email="to.liujn@outlook.com",
    description="SituatedThinker: Grounding LLM Reasoning with Real-World through Situated Thinking",
    install_requires=install_requires,
    package_data={"verl": ["verl/trainer/config/*.yaml"],},
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown"
)