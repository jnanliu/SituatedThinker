# SituatedThinker: Grounding LLM Reasoning with Real-World through Situated Thinking

![SituatedThinker Overview](assets/situated_thinker.png)

## 📋 Introduction
Recent advances in large language models (LLMs) demonstrate their impressive reasoning capabilities. However, the reasoning confined to internal parametric space limits LLMs' access to real-time information and understanding of the physical world.
To overcome this constraint, we introduce SituatedThinker, a novel framework that enables LLMs to ground their reasoning in real-world contexts through _situated thinking_, which adaptively combines both internal knowledge and external information with predefined interfaces. By utilizing reinforcement learning, SituatedThinker incentivizes deliberate reasoning with the real world to acquire information and feedback, allowing LLMs to surpass their knowledge boundaries and enhance reasoning. Experimental results demonstrate significant performance improvements on multi-hop question-answering and mathematical reasoning benchmarks. Furthermore, SituatedThinker demonstrates strong performance on unseen tasks, such as KBQA, TableQA, and text-based games, showcasing the generalizable real-world grounded reasoning capability.

## 📦 Dependencies

* 🐍 Python 3.10

## ⚙️ Installation

Run the following commands to set up the environment and install dependencies:

```bash
conda create -n situated-thinker python=3.12
conda activate situated-thinker
cd src
pip install -e .
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## 🚀 Quick Start

### 🧠 GRPO Training

#### 🔍 Prepare Retrieval Interface

##### 🗂️ Index Corpus

We utilize the Wikipedia 2018 dump as the corpus for the retrieval interface.

First, deploy the embedding model to generate embeddings for both text and queries:

```bash
scripts/deploy/deploy_llm.sh [your_embedding_model_name] [num_gpu]
```

Update the following entries in `src/runtime.env`:

```python
RETRIEVAL_EMBED_MODEL=your_embedding_model_name
RETRIEVAL_EMBED_KEY=EMPTY
RETRIEVAL_EMBED_DIM=your_embedding_model_dim
RETRIEVAL_EMBED_URL=your_embedding_model_url
```

Then, generate the FAISS index for the corpus:

```bash
scripts/build_faiss_index.sh
```

The index and corresponding corpus will be saved in `cache/faiss_index/wikipedia18`.

##### 🚀 Deploy Retrieval Interface

Deploy the retrieval interface on a machine with at least one A100-80G GPU or equivalent:

```bash
scripts/deploy/deploy_retrieval_interface.sh
```

Obtain the URL of the deployed retrieval interface.

#### 🧪 Prepare Code Execution Interface

##### 🛡️ Deploy Sandbox

Refer to [SandboxFusion](https://github.com/bytedance/SandboxFusion) to deploy the sandbox service. Then, set your sandbox service URL in `src/runtime.env`:

```python
SANDBOX_FUSION_ENDPOINT=your_sandbox_url
```

#### 📄 Prepare Training Data

Run the following script to generate training data for GRPO:

```bash
scripts/build_grpo_data.sh
```

The data will be saved in `cache/data/grpo`.

#### 🏋️‍♂️ Training

##### ⚡ Start Ray Cluster

On the master machine, start the Ray cluster:

```bash
ray start --head --port=8266
```

On other machines, connect to the master node:

```bash
ray start --address=[master_machine_ip]:8266
```

##### 🎯 Start Training

On the master machine, run the following script to initiate training:

```bash
export WANDB_KEY=your_wandb_key
export SWANLAB_API_KEY=your_swanlab_key
export RETRIEVAL_URL=your_retrieval_url

scripts/run_grpo_multinode.sh [llm_name] [num_node] [num_gpu_per_node] [tp_for_rollout] [gpu_memory_utilization_for_rollout]
```

### 🧪 Evaluation

*To be added.*

