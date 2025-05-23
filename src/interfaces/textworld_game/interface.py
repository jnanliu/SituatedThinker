from interfaces.base import BaseInterface
from interfaces.textworld_game.game import Games


class Feedback(BaseInterface):
    """
    A class that inherits from BaseInterface, designed to obtain feedback from a game.
    
    This interface takes a game ID and a sequence of commands, then returns the text 
    observation produced by the game in response to the last command.
    """
    def __init__(self):
        """
        Initialize the Feedback interface.
        
        Calls the parent class's constructor with specific parameters related to 
        feedback retrieval and initializes the game factory.
        """
        super().__init__(
            name="Obtain Feedback",  # Name of the interface
            start_tag="<feedback>",  # Start tag for the interface
            end_tag="</feedback>",  # End tag for the interface
            description="This interface returns text observation produced by the game in response to the last command given the game id and the commmand sequence.",  # Description of the interface
            query_format="game id | command1, command2, ...",  # Expected format of the query
            max_invoke_num=50  # Maximum number of times the interface can be invoked
        )
        # Initialize the game factory instance
        self.game_factory = Games.get_instance()

    async def invoke(
        self, 
        query: str,
        *args, 
        **kwargs
    ) -> str:
        """
        Asynchronously invoke the feedback retrieval operation.

        Args:
            query (str): A string containing the game ID and command sequence, separated by '|'.
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).

        Returns:
            tuple: A tuple containing an integer status code (1 for success, 0 for failure) 
                   and either the game feedback or an exception.
        """
        try:
            # Split the query string into game ID and command sequence
            game_id, commands = query.split("|")
            # Remove leading and trailing whitespace from game ID and commands
            game_id, commands = game_id.strip(), commands.strip()
        except:
            # Return failure status and error message if query format is invalid
            return 0, "Invalid query format. Please use the format <feedback>game id | command1, command2, ...</feedback>."
        try:
            # Convert the game ID to an integer to validate it
            int(game_id)
        except:
            # Return failure status and error message if game ID is invalid
            return 0, "Invalid game id. Please use an integer."
        try:
            # Asynchronously retrieve the game feedback
            feedback = await self.game_factory.get_feedback(game_id, commands)
            # Set status code to 1 indicating success
            status = 1
        except Exception as e:
            # Capture the exception if retrieval fails
            feedback = e
            # Set status code to 0 indicating failure
            status = 0
        return status, feedback

class Description(BaseInterface):
    """
    A class that inherits from BaseInterface, designed to obtain the description of the current game room.
    
    This interface takes a game ID and a sequence of commands, then returns the text 
    description of the current room.
    """
    def __init__(self):
        """
        Initialize the Description interface.
        
        Calls the parent class's constructor with specific parameters related to 
        room description retrieval and initializes the game factory.
        """
        super().__init__(
            name="Obtain Description",  # Name of the interface
            start_tag="<description>",  # Start tag for the interface
            end_tag="</description>",  # End tag for the interface
            description="This interface returns text description of the current room given game id and the commmand sequence in the query format.",  # Description of the interface
            query_format="game id | command1, command2, ...",  # Expected format of the query
            max_invoke_num=50  # Maximum number of times the interface can be invoked
        )
        # Initialize the game factory instance
        self.game_factory = Games.get_instance()

    async def invoke(
        self, 
        query: str,
        *args, 
        **kwargs
    ) -> str:
        """
        Asynchronously invoke the room description retrieval operation.

        Args:
            query (str): A string containing the game ID and command sequence, separated by '|'.
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).

        Returns:
            tuple: A tuple containing an integer status code (1 for success, 0 for failure) 
                   and either the room description or an exception.
        """
        try:
            # Split the query string into game ID and command sequence
            game_id, commands = query.split("|")
            # Remove leading and trailing whitespace from game ID and commands
            game_id, commands = game_id.strip(), commands.strip()
        except:
            # Return failure status and error message if query format is invalid
            return 0, "Invalid query format. Please use the format <description>game id | command1, command2, ... </description>."
        try:
            # Convert the game ID to an integer to validate it
            int(game_id)
        except:
            # Return failure status and error message if game ID is invalid
            return 0, "Invalid game id. Please use an integer."
        try:
            # Asynchronously retrieve the room description
            feedback = await self.game_factory.get_description(game_id, commands)
            # Set status code to 1 indicating success
            status = 1
        except Exception as e:
            # Capture the exception if retrieval fails
            feedback = e
            # Set status code to 0 indicating failure
            status = 0
        return status, feedback
 
class AdmissibleCommands(BaseInterface):
    """
    A class that inherits from BaseInterface, designed to obtain all admissible commands for the current game state.
    
    This interface takes a game ID and a sequence of commands, then returns all commands 
    relevant to the current state.
    """
    def __init__(self):
        """
        Initialize the AdmissibleCommands interface.
        
        Calls the parent class's constructor with specific parameters related to 
        admissible commands retrieval and initializes the game factory.
        """
        super().__init__(
            name="Obtain Admissible Commands",  # Name of the interface
            start_tag="<admissiblecommand>",  # Start tag for the interface
            end_tag="</admissiblecommand>",  # End tag for the interface
            description="This interface returns all commands relevant to the current state given game id and the commmand sequence in the query format.",  # Description of the interface
            query_format="game id | command1, command2, ...",  # Expected format of the query
            max_invoke_num=50  # Maximum number of times the interface can be invoked
        )
        # Initialize the game factory instance
        self.game_factory = Games.get_instance()

    async def invoke(
        self, 
        query: str,
        *args, 
        **kwargs
    ) -> str:
        """
        Asynchronously invoke the admissible commands retrieval operation.

        Args:
            query (str): A string containing the game ID and command sequence, separated by '|'.
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).

        Returns:
            tuple: A tuple containing an integer status code (1 for success, 0 for failure) 
                   and either the admissible commands or an exception.
        """
        try:
            # Split the query string into game ID and command sequence
            game_id, commands = query.split("|")
            # Remove leading and trailing whitespace from game ID and commands
            game_id, commands = game_id.strip(), commands.strip()
        except:
            # Return failure status and error message if query format is invalid
            return 0, "Invalid query format. Please use the format <admissiblecommand>game id | command1, command2, ... </dadmissiblecommand>."
        try:
            # Convert the game ID to an integer to validate it
            int(game_id)
        except:
            # Return failure status and error message if game ID is invalid
            return 0, "Invalid game id. Please use an integer."
        try:
            # Asynchronously retrieve the admissible commands
            feedback = await self.game_factory.get_admissible_commands(game_id, commands)
            # Set status code to 1 indicating success
            status = 1
        except Exception as e:
            # Capture the exception if retrieval fails
            feedback = e
            # Set status code to 0 indicating failure
            status = 0
        return status, feedback

class PossibleAdmissibleCommands(BaseInterface):
    """
    A class that inherits from BaseInterface, designed to obtain all possible admissible commands for a game.
    
    This interface takes a game ID and returns all possible commands for that game.
    """
    def __init__(self):
        """
        Initialize the PossibleAdmissibleCommands interface.
        
        Calls the parent class's constructor with specific parameters related to 
        possible admissible commands retrieval and initializes the game factory.
        """
        super().__init__(
            name="Obtain Possible Admissible Commands",  # Name of the interface
            start_tag="<possibleadmissiblecommand>",  # Start tag for the interface
            end_tag="</possibleadmissiblecommand>",  # End tag for the interface
            description="This interface returns all possible commands given game id.",  # Description of the interface
            query_format="game id",  # Expected format of the query
            max_invoke_num=50  # Maximum number of times the interface can be invoked
        )
        # Initialize the game factory instance
        self.game_factory = Games.get_instance()

    async def invoke(
        self, 
        query: str,
        *args, 
        **kwargs
    ) -> str:
        """
        Asynchronously invoke the possible admissible commands retrieval operation.

        Args:
            query (str): A string containing the game ID.
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).

        Returns:
            tuple: A tuple containing an integer status code (1 for success, 0 for failure) 
                   and either the possible admissible commands or an exception.
        """
        try:
            # Extract the game ID from the query and remove leading and trailing whitespace
            game_id = query
            game_id = game_id.strip()
        except:
            # Return failure status and error message if query format is invalid
            return 0, "Invalid query format. Please use the format <possibleadmissiblecommand>game id</possibledadmissiblecommand>."
        try:
            # Convert the game ID to an integer to validate it
            int(game_id)
        except:
            # Return failure status and error message if game ID is invalid
            return 0, "Invalid game id. Please use an integer."
        try:
            # Asynchronously retrieve the possible admissible commands
            feedback = await self.game_factory.get_possible_admissible_commands(game_id)
            # Set status code to 1 indicating success
            status = 1
        except Exception as e:
            # Capture the exception if retrieval fails
            feedback = e
            # Set status code to 0 indicating failure
            status = 0
        return status, feedback
