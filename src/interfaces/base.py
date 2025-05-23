import re
from typing import Union, Dict, List, Tuple
from abc import abstractmethod

import httpx
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv("runtime.env"))


BASE_TEMPLATE = """**Interface For {name}**

- **Description:** {description}

- **Query Format:** {start_tag} {query_format} {end_tag}.

- **Invoke Limit** {invoke_limit}."""


class BaseInterface:
    def __init__(
        self,
        name: str,
        start_tag: str,
        end_tag: str,
        description: str,
        query_format: str,
        max_invoke_num: int,
        max_retry: int = 3
    ):
        self.name = name
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.description = description
        self.query_format = query_format
        self.max_invoke_num = max_invoke_num
        self.max_retry = max_retry
 
        self.prompt = BASE_TEMPLATE.format(
            name=name, 
            start_tag=start_tag, 
            end_tag=end_tag,
            description=description,
            query_format=query_format,
            invoke_limit=max_invoke_num
        )

        self.client = httpx.AsyncClient()
    
    def extract_query(self, text: str) -> Union[str, None]:
        pattern = re.escape(self.start_tag) + r"(.*?)" + re.escape(self.end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL | re.MULTILINE)
        if len(matches) == 0:
            return None
        else:
            return matches[-1].strip()
        
    def count_query(self, text: str) -> int:
        pattern = re.escape(self.start_tag) + r"(.*?)" + re.escape(self.end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL | re.MULTILINE)
        return len(matches)

    async def fetch(self, url: str, data: Dict[str, str]) -> str:
        async with self.semphare:
            retry = 0
            while retry < self.max_retry:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(url, json=data, timeout=30)
                        response = response.json()
                        if not("timed out after" in response["content"] and response["status"] == 0):
                            return response["status"], response["content"]
                        retry += 1
                        if retry >= self.max_retry:
                            return 0, f"Error when fetching {url}: Failed to fetch content within allowed retries"
                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    retry += 1
                    if retry >= self.max_retry:
                        return 0, f"Error when fetching {url}: Failed to fetch content within allowed retries"
                except Exception as e:
                    return 0, f"Error when fetching {url}: {e}"
            
            return 0, f"Error when fetching {url}: Failed to fetch content within allowed retries"

    @abstractmethod
    async def invoke(
        self, 
        query: str,
        *args, 
        **kwargs
    ) -> str:
        raise NotImplementedError
    
class InterfaceZoo(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dict = {}

    @classmethod
    def from_interface_cls_list(cls, interface_cls_list: List[BaseInterface]) -> "InterfaceZoo":
        obj = cls()
        for interface_cls in interface_cls_list:
            obj.register(interface_cls)
        return obj

    def register(self, interface_cls: BaseInterface) -> None:
        interface_obj = interface_cls()
        self[interface_obj.end_tag] = interface_obj

    @property
    def start_tags(self) -> List[str]:
        return [interface.start_tag for interface in self.interfaces]

    @property
    def end_tags(self) -> List[str]:
        return [interface.end_tag for interface in self.interfaces]

    def get_interface_by_end_tag(self, end_tag: str) -> BaseInterface:
        return self[end_tag]
    
    def count_query(self, text: str) -> Dict[str, int]:
        count = {}
        for interface in self.interfaces:
            count["_".join(interface.name.lower().split(" "))] = interface.count_query(text)
        return count

    def __setitem__(self, key, value) -> None:
        self._dict[key] = value

    def __getitem__(self, key) -> BaseInterface:
        return self._dict[key]

    def __contains__(self, key) -> bool:
        return key in self._dict

    def __str__(self) -> str:
        return str(self._dict)

    def keys(self) -> List[str]:
        return self._dict.keys()

    def values(self) -> List[str]:
        return self._dict.values()

    def items(self) -> List[Tuple[str, BaseInterface]]:
        return self._dict.items()

    @property
    def interfaces(self) -> List[BaseInterface]:
        return list(self._dict.values())