import asyncio
from collections import defaultdict
import threading

from datasets import load_dataset


class Table:
    # Class-level variable to store the single instance of the Table class (Singleton pattern)
    _instance = None
    # Threading lock to ensure thread-safety when creating the singleton instance
    _lock = threading.Lock()

    def __init__(self):
        # Synchronously run the asynchronous load_tables method to load all tables
        self.tables = asyncio.run(self.load_tables())

    @classmethod
    def get_instance(cls, *args, **kwargs) -> 'Table':
        """
        Get the singleton instance of the Table class.
        
        This method uses the double-checked locking mechanism to ensure
        that only one instance of the Table class is created across all threads.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Table: The singleton instance of the Table class.
        """
        # First check without acquiring the lock
        if cls._instance is None:
            # Acquire the lock to ensure thread-safety
            with cls._lock:
                # Second check after acquiring the lock
                if cls._instance is None:
                    cls._instance = cls(*args, **kwargs)
        return cls._instance

    async def load_tables(self) -> list[tuple[str, str, str]]:
        """
        Asynchronously load tables from the TableQAKit/WTQ dataset.

        Returns:
            defaultdict: A dictionary mapping table IDs to table data.
        """
        # Initialize a defaultdict to store all tables
        all_table = defaultdict()
        def load_split(split: str) -> None:
            """
            Load a specific split of the dataset.

            Args:
                split (str): The split of the dataset to load (e.g., "test").
            """
            # Load the specified split of the dataset
            dataset = load_dataset("TableQAKit/WTQ", split=split)
            for example in dataset:
                # Store each table in the all_table dictionary using its ID
                all_table[example["id"]] = example["table"]
        # Iterate over the specified dataset splits
        for split in ["test"]:
            # Run the load_split function in a separate thread asynchronously
            load_split(split)
        return all_table
    
    async def get_headers(self, id: str) -> list[str]:
        """
        Asynchronously get the headers of a table given its ID.

        Args:
            id (str): The ID of the table.

        Returns:
            list[str]: A list of header names.
        """
        return self.tables[id]["header"]

    async def get_column(self, id: str, header: str) -> list[str]:
        """
        Asynchronously get a column from a table given its ID and header name.

        Args:
            id (str): The ID of the table.
            header (str): The name of the column header.

        Returns:
            list[str]: A list of values in the specified column.

        Raises:
            ValueError: If the header is not found in the table.
        """
        if header not in await self.get_headers(id):
            raise ValueError(f"Header {header} not in table {id}")
        # Get the index of the header in the list of headers
        idx = (await self.get_headers(id)).index(header)
        return [row[idx] for row in self.tables[id]["rows"]]
    
    async def get_row(self, id: str, row_idx: int) -> list[str]:
        """
        Asynchronously get a row from a table given its ID and row index.

        Args:
            id (str): The ID of the table.
            row_idx (int): The index of the row.

        Returns:
            list[str]: A list of values in the specified row.

        Raises:
            ValueError: If the row index is out of range.
        """
        if row_idx >= len(self.tables[id]["rows"]):
            raise ValueError(f"Row index {row_idx} out of range for table {id}")
        return self.tables[id]["rows"][row_idx]