import os
from pathlib import Path

import faiss

from interfaces.retrieval.embedding_model import VLLMAPIEmbedding


def load_index():
    # Initialize a list of embedding models for retrieval. Each model corresponds to a different URL
    # retrieved from the environment variable RETRIEVAL_EMBED_URL, split by comma.
    embed_model = [
        VLLMAPIEmbedding(
            # Get the retrieval embedding model name from environment variable
            os.getenv("RETRIEVAL_EMBED_MODEL"),
            # Get the retrieval embedding API key from environment variable
            os.getenv("RETRIEVAL_EMBED_KEY"),
            url,
            # Set the query prompt for the embedding model
            query_prompt="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
        ) for url in os.getenv("RETRIEVAL_EMBED_URL").split(",")
    ]
    # Get the base path by navigating three levels up from the current file's directory
    base_path = Path(__file__).parents[3]
    
    # Create options for cloning the Faiss index to multiple GPUs
    co = faiss.GpuMultipleClonerOptions()
    # Use float16 data type to reduce memory usage
    co.useFloat16 = True
    # Shard the index across multiple GPUs
    co.shard = True

    # Read the Faiss index from the specified file path
    index = faiss.read_index(str(base_path.joinpath('cache', 'faiss_index', os.getenv("RETRIEVAL_CORPUS_NAME"), 'index.bin')))
    # Initialize an empty list to store the corpus
    corpus = []
    # Open the corpus text file and read its contents line by line
    with open(base_path.joinpath('cache', 'faiss_index', os.getenv("RETRIEVAL_CORPUS_NAME"), 'corpus.txt'), 'r') as f:
        for line in f:
            # Strip whitespace from each line and append it to the corpus list
            corpus.append(line.strip())
    # Create a dictionary containing the GPU-cloned Faiss index and the corpus
    index_dict = {
        "index": faiss.index_cpu_to_all_gpus(index, co=co),
        "corpus": corpus
    }
    # Return the index dictionary and the list of embedding models
    return index_dict, embed_model