import os
from typing import List
import random

import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv("runtime.env"))

from interfaces.retrieval.index_loader import load_index


class Input(BaseModel):
    query: str | List[str]
    top_k: int

class Output(BaseModel):
    status: int
    content: str | List[str]


app = FastAPI(title='search')
index_dict, embed_model = load_index()

@app.post('/', response_model=Output)
def search(input: Input):
    """
    Perform a search operation based on the input query and return the top-k relevant documents.

    Args:
        input (Input): An instance of the Input model containing the query and the number of top documents to retrieve.

    Returns:
        Output: An instance of the Output model containing the status of the operation and the retrieved documents.
    """
    try:
        # Retrieve the Faiss index from the index dictionary
        index: faiss.Index = index_dict["index"]

        if isinstance(input.query, str):
            # If the query is a single string, get its embedding using a randomly selected embedding model
            query_embedding = random.choice(embed_model).get_query_embedding(input.query)
            # Search the index for the top-k nearest neighbors of the query embedding
            D, I = index.search(np.array([query_embedding]), input.top_k)
            # Retrieve the corresponding documents from the corpus based on the indices
            docs = [index_dict["corpus"][idx] for idx in I[0][:input.top_k]]
            # Remove duplicate documents
            docs = list(set(docs))
            # Format the documents with a prefix indicating their order
            docs = [f"doc{i}: {doc}" for i, doc in enumerate(docs)]
            # Join the formatted documents with double line breaks
            response = "\n\n".join(docs)

            return Output(
                status=1,  # Indicate that the operation was successful
                content=response
            )
        else:
            # If the query is a list of strings, get their embeddings using a randomly selected embedding model
            query_embedding = random.choice(embed_model).get_query_embeddings(input.query)
            # Search the index for the top-k nearest neighbors of each query embedding
            D, I = index.search(np.array(query_embedding), input.top_k)
            responses = []
            for idxs in I:
                # Retrieve the corresponding documents from the corpus based on the indices
                docs = [index_dict["corpus"][idx] for idx in idxs[:input.top_k]]
                # Remove duplicate documents
                docs = list(set(docs))
                # Format the documents with a prefix indicating their order
                docs = [f"doc{i}: {doc}" for i, doc in enumerate(docs)]
                # Join the formatted documents with double line breaks
                response = "\n\n".join(docs)
                responses.append(response)

            return Output(
                status=1,  # Indicate that the operation was successful
                content=responses
            )
    except Exception as e:
        return Output(
            status=0,  # Indicate that an error occurred during the operation
            content=f"Error when perform searching: {e}"
        )
        

if __name__ == "__main__":
    uvicorn.run("interfaces.retrieval.api:app", host="0.0.0.0", port=8000, ws_max_queue=4096, timeout_keep_alive=60)
