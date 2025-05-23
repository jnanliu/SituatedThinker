from typing import Any, List

from vllm import LLM
from openai import OpenAI, AsyncOpenAI

    
class VLLMEmbedding:
    _model: LLM = None
    _query_prompt: str = None
    _text_prompt: str = None

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        query_prompt: str = "",
        text_prompt: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._query_prompt = query_prompt
        self._text_prompt = text_prompt
        self._model = LLM(
            model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return f"vllm_embedding"

    def get_query_embedding(self, query: str) -> List[float]:
        return self._embed([self._query_prompt + query])[0]
    
    def get_query_embeddings(self, queries: List[str]) -> List[float]:
        return self._embed([self._query_prompt + query for query in queries])

    def get_text_embedding(self, text: str) -> List[float]:
        return self._embed([self._text_prompt + text])[0]
    
    def get_text_embeddings(self, texts: List[str]) -> List[float]:
        return self._embed([self._text_prompt + text for text in texts])
    
    def _embed(self, texts: List[str], use_tqdm: bool = False) -> List[List[float]]:
        outputs = self._model.embed(texts, use_tqdm=use_tqdm)
        embeds = []
        for output in outputs:
            embeds.append(output.outputs.embedding)
        return embeds
    
class VLLMAPIEmbedding:
    _client: OpenAI = None
    _aclient: AsyncOpenAI = None
    _path: str = None
    _query_prompt: str = None
    _text_prompt: str = None

    def __init__(
        self,
        path: str,
        api_key: str,
        api_base: str,
        query_prompt: str = "",
        text_prompt: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._path = path
        self._query_prompt = query_prompt
        self._text_prompt = text_prompt
        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self._aclient = AsyncOpenAI(api_key=api_key, base_url=api_base)

    @classmethod
    def class_name(cls) -> str:
        return f"vllm_api_embedding"

    def _embed(self, texts: List[str]) -> List[List[float]]:
        responses = self._client.embeddings.create(
            input=texts,
            model=self._path
        )
        embeds = []
        for data in responses.data:
            embeds.append(data.embedding)
        return embeds
    
    async def _async_embed(self, texts: List[str]) -> List[List[float]]:
        responses = await self._aclient.embeddings.create(
            input=texts,
            model=self._path
        )
        embeds = []
        for data in responses.data:
            embeds.append(data.embedding)
        return embeds
    
    def get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
    
    async def async_get_text_embedding(self, text: str) -> List[float]:
        embed = await self._async_get_text_embedding(text)
        return embed
    
    def get_text_embeddings(self, texts: List[str]) -> List[float]:
        return self._embed([self._text_prompt + text for text in texts])
    
    async def async_get_text_embeddings(self, texts: str) -> List[float]:
        embeds = await self._async_embed([self._text_prompt + text for text in texts])
        return embeds

    def get_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
    
    async def async_get_query_embedding(self, query: str) -> List[float]:
        embed = await self._async_get_query_embedding(query)
        return embed
    
    def get_query_embeddings(self, queries: List[str]) -> List[float]:
        return self._embed([self._query_prompt + query for query in queries])
    
    async def async_get_query_embeddings(self, queries: str) -> List[float]:
        embeds = await self._async_embed([self._query_prompt + query for query in queries])
        return embeds

    async def _async_get_query_embedding(self, query: str) -> List[float]:
        embeds = await self._async_embed([self._query_prompt + query])
        return embeds[0]

    async def _async_get_text_embedding(self, text: str) -> List[float]:
        embeds = await self._async_embed([self._text_prompt + text])
        return embeds[0]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([self._query_prompt + query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([self._text_prompt + text])[0]
