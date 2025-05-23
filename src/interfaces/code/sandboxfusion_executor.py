import traceback

from sandbox_fusion import run_code, run_code_async, RunCodeRequest, RunCodeResponse, RunStatus


async def code_executor(code_snippet: str) -> str:
    """
    Asynchronously execute a given Python code snippet using the sandbox fusion service.
    If the last line of the code does not contain a 'print' statement, 
    it modifies the code to print the result of the last line.

    Args:
        code_snippet (str): The Python code snippet to be executed.

    Returns:
        tuple: A tuple containing an execution status code (1 for success, 0 for failure) 
               and the output or error message.
    """
    try:
        # Split the code snippet into lines
        code = code_snippet.split("\n")
        # Check if the last line of the code contains a 'print' statement
        if "print(" not in code[-1]:
            # If not, modify the code to assign the result of the last line to 'answer' and print it
            code_snippet = "\n".join(code[:-1] + ["answer = " + code[-1], "print(answer)"])
        else:
            # If it does, keep the original code
            code_snippet = "\n".join(code)

        # Asynchronously execute the code using the sandbox fusion service
        output: RunCodeResponse = await run_code_async(
            RunCodeRequest(code=code_snippet, language="python"), 
            max_attempts=5,
            client_timeout=5
        )
        # Check if the code execution was successful
        if output.run_result.return_code == 0:
            # Return success status and standard output if execution was successful
            return (1, output.run_result.stdout)
        else:
            # If execution failed, get the last non-empty line of the standard error
            return (0, list(filter(lambda x: x != "", output.run_result.stderr.split("\n")[::-1]))[0])
    except Exception as e:
        # If an exception occurs during execution, return failure status and error message
        error_msg = "\n".join(traceback.format_exc().split("\n")[-3:])
        output = (0, f"Error from code executor: {error_msg}")
        return output