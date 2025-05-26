import os
import traceback
import asyncio
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv("runtime.env"))

from interfaces.base import BaseInterface
from interfaces.code import sandboxfusion_executor


executors = {
    "sandbox_fusion": sandboxfusion_executor.code_executor
}

class CodeExecution(BaseInterface):
    def __init__(self):
        super().__init__(
            name="Code Execution",
            start_tag="<code>",
            end_tag="</code>",
            description="This interface executes provided Python code snippets and returns the results, making it suitable for tasks such as data processing, analysis, computation, and validation.",
            query_format="code snippets",
            max_invoke_num=5
        )
        self.semphare = asyncio.Semaphore(32)

    async def invoke(
        self, 
        query: str,
        *args, 
        **kwargs
    ) -> str:
        """
        Asynchronously invokes the code execution process. Extracts the code snippet from the query,
        executes it using the specified code executor, and returns the execution result.

        Args:
            query (str): The input data containing the Python code snippet to be executed.
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).

        Returns:
            tuple: A tuple containing the execution status (1 for success, 0 for failure) 
                   and the execution result or error message.
        """
        try:
            # Extract the code snippet from the input data
            # Currently, the entire query is considered as the code snippet
            code_snippet = query
            # Execute the code snippet using the code_executor function
            # The executor is determined by the environment variable CODE_EXECUTION_BACKEND
            async with self.semphare:
                status, result = await executors[os.getenv("CODE_EXECUTION_BACKEND")](code_snippet)
            # Return the execution result as a tuple of status and result
            return status, result
        except Exception as e:
            # If an error occurs during the code execution, capture the last 3 lines of the traceback
            error_msg = "\n".join(traceback.format_exc().split("\n")[-3:])
            # Return a failure status and the error message
            return 0, f"Error when code execution: {error_msg}"