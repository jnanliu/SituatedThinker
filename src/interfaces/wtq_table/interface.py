from interfaces.base import BaseInterface
from interfaces.wtq_table.table import Table


class Headers(BaseInterface):
    """
    A class that inherits from BaseInterface, designed to retrieve the headers of a table.
    """
    def __init__(self):
        """
        Initialize the Headers interface.
        
        Calls the parent class's constructor with specific parameters related to 
        header retrieval and initializes the table interface.
        """
        super().__init__(
            name="Header Retrieval",  # Name of the interface
            start_tag="<header>",  # Start tag for the interface
            end_tag="</header>",  # End tag for the interface
            description="This interface retrieves the headers of the table specified by the given table.",  # Description of the interface
            query_format="table id",  # Expected format of the query
            max_invoke_num=10  # Maximum number of times the interface can be invoked
        )
        # Initialize the table interface instance
        self.table_interface = Table.get_instance()

    async def invoke(
        self, 
        query: str,
        *args, 
        **kwargs
    ) -> str:
        """
        Asynchronously invoke the header retrieval operation.

        Args:
            query (str): The ID of the table for which headers are to be retrieved.
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).

        Returns:
            tuple: A tuple containing an integer status code (1 for success, 0 for failure) 
                   and either the string representation of table headers or an exception.
        """
        try:
            # Asynchronously retrieve the headers of the specified table
            headers = await self.table_interface.get_headers(query)
            # Convert the headers to a string
            headers = str(headers)
            # Set status code to 1 indicating success
            status = 1
        except Exception as e:
            # Capture the exception if retrieval fails
            headers = e
            # Set status code to 0 indicating failure
            status = 0
        return status, headers

class Column(BaseInterface):
    """
    A class that inherits from BaseInterface, designed to retrieve a column from a table.
    """
    def __init__(self):
        """
        Initialize the Column interface.
        
        Calls the parent class's constructor with specific parameters related to 
        column retrieval and initializes the table interface.
        """
        super().__init__(
            name="Column Retrieval",  # Name of the interface
            start_tag="<column>",  # Start tag for the interface
            end_tag="</column>",  # End tag for the interface
            description="This interface retrieves a column of the table specified by the given table id and header.",  # Description of the interface
            query_format="table id, header name",  # Expected format of the query
            max_invoke_num=10  # Maximum number of times the interface can be invoked
        )
        # Initialize the table interface instance
        self.table_interface = Table.get_instance()

    async def invoke(
        self, 
        query: str,
        *args, 
        **kwargs
    ) -> str:
        """
        Asynchronously invoke the column retrieval operation.

        Args:
            query (str): A string containing the table ID and header name, separated by a comma.
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).

        Returns:
            tuple: A tuple containing an integer status code (1 for success, 0 for failure) 
                   and either the string representation of a table column or an exception.
        """
        try:
            # Split the query string into table ID and header name
            table_id, header_name = query.split(",")
            # Remove leading and trailing whitespace from table ID and header name
            table_id, header_name = table_id.strip(), header_name.strip()
        except:
            # Return failure status and error message if query format is invalid
            return 0, "Invalid query format. Please use the format <column>table id, header name</column>."
        
        try:
            # Asynchronously retrieve the specified column from the table
            column = await self.table_interface.get_column(table_id, header_name)
            # Convert the column to a string
            column = str(column)
            # Set status code to 1 indicating success
            status = 1
        except Exception as e:
            # Capture the exception if retrieval fails
            column = e
            # Set status code to 0 indicating failure
            status = 0
        return status, column
    
class Row(BaseInterface):
    """
    A class that inherits from BaseInterface, designed to retrieve a row from a table.
    """
    def __init__(self):
        """
        Initialize the Row interface.
        
        Calls the parent class's constructor with specific parameters related to 
        row retrieval and initializes the table interface.
        """
        super().__init__(
            name="Row Retrieval",  # Name of the interface
            start_tag="<row>",  # Start tag for the interface
            end_tag="</row>",  # End tag for the interface
            description="This interface retrieves a row of the table specified by the given table id and row index.",  # Description of the interface
            query_format="table id, row index",  # Expected format of the query
            max_invoke_num=10  # Maximum number of times the interface can be invoked
        )
        # Initialize the table interface instance
        self.table_interface = Table.get_instance()

    async def invoke(
        self, 
        query: str,
        *args, 
        **kwargs
    ) -> str:
        """
        Asynchronously invoke the row retrieval operation.

        Args:
            query (str): A string containing the table ID and row index, separated by a comma.
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).

        Returns:
            tuple: A tuple containing an integer status code (1 for success, 0 for failure) 
                   and either the string representation of a table row or an exception.
        """
        try:
            # Split the query string into table ID and row index
            table_id, row_indice = query.split(",")
            # Remove leading and trailing whitespace from table ID and row index
            table_id, row_indice = table_id.strip(), row_indice.strip()
        except:
            # Return failure status and error message if query format is invalid
            return 0, "Invalid query format. Please use the format <row>table id, row index</row>."
        
        try:
            # Convert the row index to an integer
            row_indice = int(row_indice)
        except:
            # Return failure status and error message if row index is invalid
            return 0, "Invalid row index. Please use an integer."
        
        try:
            # Asynchronously retrieve the specified row from the table
            row = await self.table_interface.get_row(table_id, row_indice)
            # Convert the row to a string
            row = str(row)
            # Set status code to 1 indicating success
            status = 1
        except Exception as e:
            # Capture the exception if retrieval fails
            row = e
            # Set status code to 0 indicating failure
            status = 0
        return status, row
