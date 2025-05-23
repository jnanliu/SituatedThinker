from interfaces.base import BaseInterface
from interfaces.webqsp_freebase.knowledge_graph import KnowledgeGraph


class RelationRetrieval(BaseInterface):
    """
    A class that inherits from BaseInterface, designed to retrieve neighboring relations 
    for a given entity from a knowledge graph.
    """
    def __init__(self):
        """
        Initialize the RelationRetrieval instance.
        
        Calls the parent class's constructor with specific parameters and 
        initializes the knowledge graph interface.
        """
        super().__init__(
            name="Relation Retrieval",
            start_tag="<relation>",
            end_tag="</relation>",
            description="This interface retrieves the neighboring relations given the entity.",
            query_format="entity",
            max_invoke_num=10
        )
        # Initialize the knowledge graph interface instance
        self.kg_interface = KnowledgeGraph.get_instance()

    async def invoke(
        self, 
        query: str, 
        *args, 
        **kwargs
    ) -> str:
        """
        Asynchronously retrieve neighboring relations for a given entity.

        Args:
            query (str): The entity for which to retrieve neighboring relations.
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).

        Returns:
            tuple: A tuple containing an integer status code (1 for success, 0 for failure) 
                   and either the string representation of neighboring relations or an exception.
        """
        try:
            # Asynchronously retrieve neighboring relations from the knowledge graph
            neighbor_relations = await self.kg_interface.get_neighbor_relations(query)
            # Convert the result to a string
            neighbor_relations = str(neighbor_relations)
            # Set status code to 1 indicating success
            status = 1
        except Exception as e:
            # Capture the exception if retrieval fails
            neighbor_relations = e
            # Set status code to 0 indicating failure
            status = 0
        return status, neighbor_relations

class TailEntityRetrieval(BaseInterface):
    """
    A class that inherits from BaseInterface, designed to retrieve tail entities 
    associated with a given head entity and relation from a knowledge graph.
    """
    def __init__(self):
        """
        Initialize the TailEntityRetrieval instance.
        
        Calls the parent class's constructor with specific parameters and 
        initializes the knowledge graph interface.
        """
        super().__init__(
            name="Tail Entity Retrieval",
            start_tag="<entity>",
            end_tag="</entity>",
            description="This interface retrieves the tail entities associated with a given head entity and relation.",
            query_format="head entity, relation",
            max_invoke_num=10
        )
        # Initialize the knowledge graph interface instance
        self.kg_interface = KnowledgeGraph.get_instance()

    async def invoke(
        self, 
        query: str, 
        *args, 
        **kwargs
    ) -> str:
        """
        Asynchronously retrieve tail entities for a given head entity and relation.

        Args:
            query (str): A string containing the head entity and relation, separated by a comma.
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).

        Returns:
            tuple: A tuple containing an integer status code (1 for success, 0 for failure) 
                   and either the string representation of tail entities or an exception.
        """
        try:
            # Split the query string into head entity and relation
            head_entity, relation = query.split(",")
            # Remove leading and trailing whitespace from head entity and relation
            head_entity, relation = head_entity.strip(), relation.strip()
        except:
            # Return failure status and error message if query format is invalid
            return 0, "Invalid query format. Please use the format <entity>head entity, relation</entity>."
        try:
            # Asynchronously retrieve tail entities from the knowledge graph
            tail_entities = await self.kg_interface.get_tail_entities(head_entity, relation)
            # Convert the result to a string
            tail_entities = str(tail_entities)
            # Set status code to 1 indicating success
            status = 1
        except Exception as e:
            # Capture the exception if retrieval fails
            tail_entities = e
            # Set status code to 0 indicating failure
            status = 0
        # Return the status code and the result or exception
        return status, tail_entities