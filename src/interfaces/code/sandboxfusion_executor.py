import traceback

import asyncio
import aiohttp
import logging
from typing import List, Optional

from tenacity import RetryCallState
from sandbox_fusion.common import trim_slash
from sandbox_fusion.client import wraps, retry, wait_exponential_jitter, stop_after_attempt, before_retry_sleep
from sandbox_fusion import config

from sandbox_fusion.models import RunCodeRequest, RunCodeResponse, RunStatus

logger = logging.getLogger(__name__)


def on_retry_error(s):
    e = s.outcome.exception()
    error_msg = "\n".join(traceback.format_exc().split('\n')[-3:])
    logger.error(f'give up requesting sandbox. error: {error_msg}')
    raise e

def configurable_retry(max_attempts):

    def decorator(func):

        @wraps(func)
        @retry(wait=wait_exponential_jitter(),
               stop=stop_after_attempt(max_attempts),
               before_sleep=before_retry_sleep,
               retry_error_callback=on_retry_error)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        @wraps(func)
        @retry(wait=wait_exponential_jitter(),
               stop=stop_after_attempt(max_attempts),
               before_sleep=before_retry_sleep,
               retry_error_callback=on_retry_error)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

async def run_code(request: RunCodeRequest,
                   endpoint: str = '',
                   max_attempts: int = 5,
                   client_timeout: Optional[float] = None) -> RunCodeResponse:

    @configurable_retry(max_attempts)
    async def _run_code(request: RunCodeRequest) -> RunCodeResponse:
        timeout = aiohttp.ClientTimeout(total=client_timeout) if client_timeout else None
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f'{trim_slash(endpoint or config.SANDBOX_ENDPOINT)}/run_code',
                                    json=request.dict()) as result:
                if result.status != 200:
                    raise Exception(f'Faas api responded with code {result.status}: {await result.text()}')
                resp = RunCodeResponse(**(await result.json()))
                if resp.status == RunStatus.SandboxError:
                    raise Exception(f'Sandbox responded with error: {resp.message}')
                return resp

    return await _run_code(request)

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
        output: RunCodeResponse = await run_code(
            RunCodeRequest(code=code_snippet, language="python"), 
            max_attempts=1,
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