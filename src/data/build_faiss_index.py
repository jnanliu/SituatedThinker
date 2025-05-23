import os
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
from datasets import DatasetDict
import faiss
from llama_index.core.node_parser import SentenceSplitter

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv("runtime.env"))

from data.dataset_loader import DATASET_INFOS, load_dataset
from interfaces.retrieval.embedding_model import VLLMAPIEmbedding


splitter = SentenceSplitter(
    chunk_size=256,
    chunk_overlap=32
)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        choices=list(DATASET_INFOS.keys())
    )
    args = parser.parse_args()
    return args

def construct_documents(datasets: DatasetDict) -> List[str]:
    """
    Construct a list of documents from a DatasetDict object.

    This function iterates through each split in the dataset and aggregates document content 
    based on the sample and document IDs. It groups the content by these IDs and combines them 
    into a single string per document.

    Args:
        datasets (DatasetDict): A dictionary-like object containing datasets for different splits.

    Returns:
        List[str]: A list of strings, where each string represents a combined document content.
    """
    documents = []
    # Iterate through each split in the dataset
    for split, dataset in datasets.items():
        # Initialize dictionaries to map IDs to titles and content
        id2title = defaultdict()
        id2content = defaultdict(list)
        # Iterate through each example in the current split
        for example in tqdm(dataset):
            # Iterate through each chunk list in the example's context
            for chunks in example['context']:
                # Iterate through each chunk in the chunk list
                for chunk in chunks:
                    # Map the combined ID to the chunk's title
                    id2title[f"{chunk['sample_id']}_{chunk['doc_id']}"] = chunk['title']
                    # Append the chunk's content to the list associated with the combined ID
                    id2content[f"{chunk['sample_id']}_{chunk['doc_id']}"].append(chunk['content'])
        # Print the number of documents in the current split
        print(f"{split}: {len(id2content)} documents")
        # Combine the content for each ID into a single document
        for id in id2content:
            documents.append("\n".join(id2content[id]))

    return documents

def split_text(text: str) -> List[str]:
    return splitter.split_text(text)


def load_embedding_model() -> List[VLLMAPIEmbedding]:
    """
    Load a list of VLLMAPIEmbedding instances based on environment variables.

    This function creates multiple VLLMAPIEmbedding objects using the model name, API key, 
    and a list of URLs specified in the environment variables. Each object is initialized 
    with a common query prompt.

    Returns:
        List[VLLMAPIEmbedding]: A list of VLLMAPIEmbedding instances.
    """
    # Create a list of VLLMAPIEmbedding instances for each URL specified in the environment variable
    models = [
        VLLMAPIEmbedding(
            # Retrieve the embedding model name from the environment variable
            os.getenv("RETRIEVAL_EMBED_MODEL"),
            # Retrieve the API key for the embedding service from the environment variable
            os.getenv("RETRIEVAL_EMBED_KEY"),
            # Use each URL from the comma-separated list in the environment variable
            url,
            # Set a common query prompt for the embedding service
            query_prompt="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        ) for url in os.getenv("RETRIEVAL_EMBED_URL").split(",")
    ]
    return models


if __name__ == "__main__":
    # Parse command - line arguments
    args = parse_args()

    # Get the base directory path by going up three levels from the current file's path
    base_path = Path(__file__).parents[2]
    # Create the directory for the Faiss index cache if it doesn't exist
    cache_path = base_path.joinpath("cache", "faiss_index", args.dataset)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Check if the corpus text file already exists
    if cache_path.joinpath("corpus.txt").exists():
        chunks = []
        # Open the corpus text file and read each line, stripping whitespace
        with open(cache_path.joinpath("corpus.txt")) as f:
            for line in f:
                chunks.append(line.strip())
        # Print the total number of chunks read from the file
        print('Total chunks: ', len(chunks))
    else:
        # Initialize a DatasetDict to hold datasets for different splits
        datasets = DatasetDict()
        # Load datasets for each split specified in DATASET_INFOS
        for split in DATASET_INFOS[args.dataset]:
            datasets[split] = load_dataset(args.dataset, split)
        # Construct a list of documents from the loaded datasets
        documents = construct_documents(datasets)
        # Print the total number of documents constructed
        print('Total documents: ', len(documents))
        idxs = []
        chunks = []
        # Use a thread pool to parallelize text splitting
        with ThreadPoolExecutor() as executor:
            tasks = [executor.submit(lambda i, text: (i, split_text(text)), i, text) for i, text in enumerate(documents)]
            # Iterate over completed tasks and collect indices and chunks
            for task in tqdm(as_completed(tasks), total=len(tasks)):
                idx, chunk = task.result()
                idxs.append(idx)
                chunks.append(chunk)
        # Sort the indices and chunks by index
        idxs, chunks = zip(*sorted(zip(idxs, chunks), key=lambda x: x[0]))
        # Flatten the list of chunks
        chunks = [chunk for chunks in chunks for chunk in chunks]
        # Print the total number of chunks after splitting
        print('Total chunks: ', len(chunks))
        # Remove newline characters from each chunk
        chunks = [chunk.replace('\n', '') for chunk in chunks]
        # Write the chunks to the corpus text file
        with open(cache_path.joinpath("corpus.txt"), 'w') as f:
            f.write('\n'.join(chunks))

    # Load the embedding model
    embedding_model = load_embedding_model()
    idxs = []
    embeddings = []
    batch_size = 256
    # Use a thread pool to parallelize embedding computation
    with ThreadPoolExecutor(len(embedding_model)) as executor:
        tasks = [
            executor.submit(lambda i, texts: (i, embedding_model[i%len(embedding_model)]._embed(texts)), i, chunks[j:j+batch_size]) 
            for i, j in enumerate(range(0, len(chunks), batch_size))
        ]
        # Iterate over completed tasks and collect indices and embeddings
        for task in tqdm(as_completed(tasks), total=len(tasks)):
            i, embed = task.result()
            idxs.append(i)
            embeddings.append(embed)
    # Sort the indices and embeddings by index
    idxs, embeddings = zip(*sorted(zip(idxs, embeddings), key=lambda x: x[0]))
    # Flatten the list of embeddings
    embeddings = [embed for embeds in embeddings for embed in embeds]
    # Print the size of the embeddings
    print("Embedding size: ", f"{len(embeddings)}x{len(embeddings[0])}")

    # Create a Faiss index with inner - product similarity
    index = faiss.IndexFlatIP(int(os.getenv("RETRIEVAL_EMBED_DIM")))
    # Add the embeddings to the Faiss index
    index.add(np.array(embeddings))
    # Write the Faiss index to a binary file
    faiss.write_index(index, cache_path.joinpath("index.bin"))
