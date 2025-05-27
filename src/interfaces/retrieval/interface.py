import os
import asyncio
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv("runtime.env"))

from interfaces.base import BaseInterface


class Retrieval(BaseInterface):
    def __init__(self):
        super().__init__(
            name="Retrieval Information",
            start_tag="<retrieval>",
            end_tag="</retrieval>",
            description="This interface retrieves the necessary information based on the given query.",
            query_format="query",
            max_invoke_num=5
        )
        self.semphare = asyncio.Semaphore(16)
        self.top_k = int(os.getenv("RETRIEVAL_TOP_K"))

    async def invoke(
        self, 
        query: str,
        *args, 
        **kwargs
    ) -> str:
        data = {
            "query": query,
            "top_k": self.top_k
        }
        status, result = await self.fetch(os.getenv("RETRIEVAL_URL"), data)
        return status, result