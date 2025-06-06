import os
import sys
import asyncio
import threading
from pathlib import Path

from textworld import gym, EnvInfos


def suppress_stdout(func):
    """
    A decorator that suppresses the standard output (stdout) of the decorated function.

    This decorator redirects the standard output to a null device during the execution of
    the decorated function, effectively suppressing any print statements or other stdout
    outputs. After the function execution is complete, the standard output is restored to
    its original state.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: A wrapper function that suppresses stdout when calling the original function.
    """
    def wrapper(*args, **kwargs):
        # Save the original standard output stream to restore it later
        original_stdout = sys.stdout
        try:
            # Open a null file object. On Unix-like systems, this is '/dev/null'; on Windows, 'nul'.
            # All data written to this file is discarded.
            with open(os.devnull, 'w') as null_file:
                # Redirect the standard output stream to the null file
                sys.stdout = null_file
                # Call the original function with the provided arguments and keyword arguments
                result = func(*args, **kwargs)
        finally:
            # Ensure that the original standard output stream is restored,
            # even if an exception occurs during the function execution
            sys.stdout = original_stdout
        return result
    return wrapper

class Games:
    # Class-level variable to store the single instance of the Games class (Singleton pattern)
    _instance = None
    # Threading lock to ensure thread-safety when creating the singleton instance
    _lock = threading.Lock()

    def __init__(self):
        # Dictionary to store environment IDs for each game
        self.env_ids = {}
        # Register 50 custom TextWorld games
        for i in range(50):
            # Register a game and get its environment ID
            env_id = gym.register_game(
                # Construct the path to the game file
                str(Path(__file__).parents[3].joinpath("cache", "data", "textworld", f"custom_game_{i}.z8")),
                # Set the maximum number of steps per episode
                max_episode_steps=100,
                # Request specific information about the game environment
                request_infos=EnvInfos(
                    description=True,  # Request the game description
                    inventory=True,  # Request the player's inventory
                    game=True,  # Request game-related information
                    facts=True,  # Request game facts
                    admissible_commands=True,  # Request admissible commands
                    possible_admissible_commands=True,  # Request possible admissible commands
                    feedback=True,  # Request feedback information
                )
            )
            # Map the game ID (as a string) to its environment ID
            self.env_ids[str(i)] = env_id

    @classmethod
    def get_instance(cls, *args, **kwargs) -> 'Games':
        """
        Get the singleton instance of the Games class.

        This method ensures that only one instance of the Games class is created.
        If the instance doesn't exist, it creates one in a thread-safe manner.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Games: The singleton instance of the Games class.
        """
        # Check if the singleton instance has not been created yet
        if cls._instance is None:
            # Acquire the lock to ensure thread-safety
            with cls._lock:
                # Double-check if the instance is still None after acquiring the lock
                if cls._instance is None:
                    # Create a new instance of the Games class
                    cls._instance = cls(*args, **kwargs)
        # Return the singleton instance
        return cls._instance
    
    @suppress_stdout
    async def get_feedback(self, game_id: str, commands: str):
        """
        Asynchronously get the feedback from a game after executing a series of commands.

        Args:
            game_id (str): The ID of the game to play.
            commands (str): A comma-separated string of commands to execute.

        Returns:
            str: The feedback from the game.
        """
        def _get_feedback():
            # Create a game environment using the provided game ID
            env = gym.make(self.env_ids[game_id]) 
            # Reset the game environment and get the initial observation and information
            obs, infos = env.reset()

            # Initialize the feedback information
            infos = {"feedback": ""}
            # Iterate over each command in the comma-separated string
            for command in commands.split(","):
                # Take a step in the game environment with the current command
                obs, score, done, infos = env.step(command)
                # Render the game environment
                env.render()
                # Check if the game is done
                if done:
                    break
            # Close the game environment
            env.close()
            # Return the feedback information
            return infos["feedback"]
        # Run the synchronous _get_feedback function in a separate thread asynchronously
        return await asyncio.to_thread(_get_feedback)
    
    @suppress_stdout
    async def get_description(self, game_id: str, commands: str):
        """
        Asynchronously get the description of a game after executing a series of commands.

        Args:
            game_id (str): The ID of the game to play.
            commands (str): A comma-separated string of commands to execute.

        Returns:
            str: The description of the game.
        """
        def _get_description():
            # Create a game environment using the provided game ID
            env = gym.make(self.env_ids[game_id]) 
            # Reset the game environment and get the initial observation and information
            obs, infos = env.reset()

            # Initialize the description information
            infos = {"description": ""}
            # Iterate over each command in the comma-separated string
            for command in commands.split(","):
                # Take a step in the game environment with the current command
                obs, score, done, infos = env.step(command)
                # Render the game environment
                env.render()
                # Check if the game is done
                if done:
                    break
            # Close the game environment
            env.close()
            # Return the description information
            return infos["description"]
        # Run the synchronous _get_description function in a separate thread asynchronously
        return await asyncio.to_thread(_get_description)
    
    @suppress_stdout
    async def get_admissible_commands(self, game_id: str, commands: str):
        """
        Asynchronously get the admissible commands of a game after executing a series of commands.

        Args:
            game_id (str): The ID of the game to play.
            commands (str): A comma-separated string of commands to execute.

        Returns:
            str: The admissible commands of the game.
        """
        def _get_admissible_commands():
            # Create a game environment using the provided game ID
            env = gym.make(self.env_ids[game_id]) 
            # Reset the game environment and get the initial observation and information
            obs, infos = env.reset()

            # Initialize the admissible commands information
            infos = {"admissible_commands": ""}
            # Iterate over each command in the comma-separated string
            for command in commands.split(","):
                # Take a step in the game environment with the current command
                obs, score, done, infos = env.step(command)
                # Render the game environment
                env.render()
                # Check if the game is done
                if done:
                    break
            # Close the game environment
            env.close()
            # Return the admissible commands information as a string
            return str(infos["admissible_commands"])
        # Run the synchronous _get_admissible_commands function in a separate thread asynchronously
        return await asyncio.to_thread(_get_admissible_commands)
    
    @suppress_stdout
    async def get_possible_admissible_commands(self, game_id: str):
        """
        Asynchronously get the possible admissible commands of a game.

        Args:
            game_id (str): The ID of the game.

        Returns:
            str: The possible admissible commands of the game.
        """
        def _get_possible_admissible_commands():
            # Create a game environment using the provided game ID
            env = gym.make(self.env_ids[game_id]) 
            # Reset the game environment and get the initial observation and information
            obs, infos = env.reset()
            # Return the possible admissible commands information as a string
            return str(infos["possible_admissible_commands"])
        # Run the synchronous _get_possible_admissible_commands function in a separate thread asynchronously
        return await asyncio.to_thread(_get_possible_admissible_commands)