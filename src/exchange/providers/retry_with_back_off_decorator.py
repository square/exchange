from functools import wraps
import time
from typing import Any, Callable, Dict, List, Optional
from httpx import HTTPStatusError, Response

def retry_with_backoff(
        should_retry: Callable, 
        max_retries: Optional[int] = 5, 
        initial_wait: Optional[int] = 10, 
        backoff_factor: Optional[int] = 1, 
        handle_retry_exhausted: Optional[Callable] = None) ->  Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: List, **kwargs: Dict) -> Any:  # noqa: ANN401
            result = None
            for retry in range(max_retries):
                result = func(*args, **kwargs)
                if not should_retry(result):
                    return result
                
                sleep_time = initial_wait + (backoff_factor * (2 ** retry))
                time.sleep(sleep_time)
            
            if should_retry(result):
                handle_retry_exhausted(result, max_retries)
        return wrapper
    return decorator

def retry_httpx_request(
        retry_on_status_code: List[int] = [429, 529, 500],
        max_retries: Optional[int] = 5, 
        initial_wait: Optional[int] = 10, 
        backoff_factor: Optional[int] = 1) -> Callable:
    
    def should_retry(response: Response) -> bool:
        return response.status_code in retry_on_status_code

    def handle_retry_exhausted(response: Response, max_retries: int) -> None:
        raise HTTPStatusError(
            f"Failed after {max_retries} retries due to rate limiting",
            request=response.request,
            response=response,
        )

    return retry_with_backoff(
        max_retries=max_retries,
        initial_wait=initial_wait,
        backoff_factor=backoff_factor,
        should_retry=should_retry,
        handle_retry_exhausted=handle_retry_exhausted  
    )