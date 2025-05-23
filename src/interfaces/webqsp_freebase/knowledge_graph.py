import asyncio
from collections import defaultdict
import threading

from datasets import load_dataset
from tqdm import tqdm


class KnowledgeGraph:
    # Class variable to hold the singleton instance of the KnowledgeGraph
    _instance = None
    # Thread lock used to ensure thread-safe singleton creation
    _lock = threading.Lock()

    def __init__(self):
        """
        Initialize the KnowledgeGraph instance.
        Loads triples asynchronously, flattens them, and populates relation and entity mappings.
        """
        # Use asyncio.run to execute the asynchronous load_triples method and flatten the result
        self.triples = self.flatten_cvt(asyncio.run(self.load_triples()))
        # Initialize a defaultdict to map head entities to their relations
        self.h2r = defaultdict(set)
        # Initialize a defaultdict to map (head entity, relation) pairs to their tail entities
        self.hr2t = defaultdict(set)
        # Iterate through all triples and populate the h2r and hr2t dictionaries
        for h, r, t in self.triples:
            self.h2r[h].add(r)
            self.hr2t[(h, r)].add(t)

    @classmethod
    def get_instance(cls, *args, **kwargs) -> 'KnowledgeGraph':
        """
        Get the singleton instance of the KnowledgeGraph.
        If the instance doesn't exist, create it in a thread-safe manner.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            KnowledgeGraph: The singleton instance of the KnowledgeGraph.
        """
        if cls._instance is None:
            # Use the lock to ensure thread-safe instance creation
            with cls._lock:
                cls._instance = cls(*args, **kwargs)
        return cls._instance

    async def load_triples(self) -> list[tuple[str, str, str]]:
        """
        Asynchronously load triples from the dataset splits (train, test, validation).

        Returns:
            list[tuple[str, str, str]]: A list of triples, where each triple is a tuple of (head, relation, tail).
        """
        # Set to store all unique triples
        all_triples = set()
        def load_split(split: str) -> None:
            """
            Load triples from a specific dataset split.

            Args:
                split (str): The dataset split to load (e.g., "train", "test", "validation").
            """
            # Load the dataset for the given split
            dataset = load_dataset("rmanluo/RoG-webqsp", split=split)
            # Iterate through each example in the dataset with a progress bar
            for example in tqdm(dataset):
                # Iterate through each triple in the example's graph
                for triple in example["graph"]:
                    # Add the triple as a tuple to the set of all triples
                    all_triples.add(tuple(triple))
        # Iterate through all dataset splits
        tasks = []
        for split in ["train", "test", "validation"]:
            # Run the load_split function in a separate thread asynchronously
            tasks.append(asyncio.create_task(asyncio.to_thread(load_split, split)))
        await asyncio.gather(*tasks)
        # Convert the set of triples to a list and return it
        return list(all_triples)
    
    def flatten_cvt(self, triples: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
        """
        Flatten the triples by expanding converted nodes.

        Args:
            triples (list[tuple[str, str, str]]): A list of triples to be flattened.

        Returns:
            list[tuple[str, str, str]]: The flattened list of triples.
        """
        # Set to store all converted nodes
        cvt_nodes: set[str] = {
            o for _, _, o in triples
            if self.is_cvt(o)
        }

        # Dictionary to map converted nodes to their properties
        cvt_props: dict[str, list[tuple[str, str]]] = {}
        # Iterate through all triples
        for subj, pred, obj in tqdm(triples):
            # Check if the subject is a converted node and the predicate is not a converted node
            if subj in cvt_nodes and not self.is_cvt(pred):
                # Add the predicate and object as a property of the converted node
                cvt_props.setdefault(subj, []).append((pred, obj))

        # List to store the flattened triples
        flattened: list[tuple[str, str, str]] = []
        # Iterate through all triples again
        for subj, pred, obj in tqdm(triples):
            if obj in cvt_nodes:
                # If the object is a converted node, expand its properties
                for p_i, o_i in cvt_props.get(obj, []):
                    flattened.append((subj, p_i, o_i))
            else:
                # If the object is not a converted node, add the original triple
                flattened.append((subj, pred, obj))

        return flattened
    
    async def get_neighbor_relations(self, h: str) -> list[str]:
        """
        Asynchronously get all neighbor relations of a given head entity.

        Args:
            h (str): The head entity.

        Returns:
            list[str]: A list of neighbor relations.
        """
        # Run the lambda function in a separate thread asynchronously to get neighbor relations
        return await asyncio.to_thread(lambda h: list(self.h2r[h]), h)

    async def get_tail_entities(self, h: str, r: str) -> list[str]:
        """
        Asynchronously get all tail entities for a given head entity and relation.

        Args:
            h (str): The head entity.
            r (str): The relation.

        Returns:
            list[str]: A list of tail entities.
        """
        # Run the lambda function in a separate thread asynchronously to get tail entities
        return await asyncio.to_thread(lambda h: list(self.hr2t[(h, r)]), h)

    def is_cvt(self, name: str) -> bool:
        """
        Check if a given name represents a converted node.

        Args:
            name (str): The name to check.

        Returns:
            bool: True if the name starts with "m.", False otherwise.
        """
        return name.startswith("m.")