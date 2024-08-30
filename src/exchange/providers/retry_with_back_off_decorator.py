from functools import wraps
import time
from typing import Any, Callable, Dict, List, Optional

from httpx import HTTPStatusError, Response


def retry_with_backoff(
        should_retry: Callable, 
        max_retries: int = 5, 
        initial_wait: int = 10, 
        backoff_factor: int =1, 
        handle_retry_exhausted: Optional[Callable] = None) ->  Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: List, **kwargs: Dict) -> Any:  # noqa: ANN401
            result = None
            for retry in range(max_retries):
                result = func(*args, **kwargs)
                
                # Use the passed should_retry function, or a default one
                if should_retry is None or not should_retry(result):
                    return result
                
                # If retry condition is met, wait and retry
                sleep_time = initial_wait + (backoff_factor * (2 ** retry))
                print(f"Retry {retry + 1}/{max_retries}: Waiting {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
            
            # Handle failure after all retries
            if result and should_retry(result):
                handle_retry_exhausted(result, max_retries)
        return wrapper
    return decorator

def should_retry(response: Response) -> bool:
    return response.status_code in (429, 529, 500)
    
def handle_retry_exhausted(response: Response, max_retries: int) -> None:
    raise HTTPStatusError(
        f"Failed after {max_retries} retries due to rate limiting",
        request=response.request,
        response=response,
    )

def retry_httpx_request(
        retry_on_status_code: List[int] = [429, 529, 500],
        max_retries: int = 5, 
        initial_wait: int = 10, 
        backoff_factor: int =1) -> Callable:
    """Wrapper decorator to pre-configure retry_with_backoff for specific status codes."""
    
    def should_retry(response: Response) -> bool:
        """Custom retry logic: Retry on specific status codes."""
        return response.status_code in retry_on_status_code

    def handle_retry_exhausted(response: Response, max_retries: int) -> None:
        """Custom failure handler."""
        print(f"Failed after {max_retries} retries with status code: {response.status_code}")
        raise RuntimeError(f"Request failed after {max_retries} retries")

    return retry_with_backoff(
        max_retries=max_retries,
        initial_wait=initial_wait,
        backoff_factor=backoff_factor,
        should_retry=should_retry,
        handle_retry_exhausted=handle_retry_exhausted  
    )