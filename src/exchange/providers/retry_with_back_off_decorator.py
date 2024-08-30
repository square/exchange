from functools import wraps
import time
from typing import Any, Callable, Dict, List


def retry_with_backoff(max_retries: int = 5, initial_wait: int = 10, backoff_factor: int =1, 
                       should_retry: Callable = None, handle_failure: Callable = None) ->  Callable:
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
            if result and (should_retry is None or should_retry(result)):
                handle_failure(result, max_retries)
        return wrapper
    return decorator