import time
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional

from httpx import HTTPStatusError, Response


def retry_with_backoff(
        should_retry: Callable, 
        max_retries: Optional[int] = 5, 
        initial_wait: Optional[float] = 10, 
        backoff_factor: Optional[float] = 1, 
        handle_retry_exhausted: Optional[Callable] = None) ->  Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: List, **kwargs: Dict) -> Any:  # noqa: ANN401
            result = None
            for retry in range(max_retries):
                result = func(*args, **kwargs)
                if not should_retry(result):
                    return result
                if (retry + 1) == max_retries:
                    break
                sleep_time = initial_wait + (backoff_factor * (2 ** retry))
                time.sleep(sleep_time)
            if handle_retry_exhausted:
                handle_retry_exhausted(result, max_retries)
            return result
        return wrapper
    return decorator

def retry_httpx_request(
    retry_on_status_code: Optional[Iterable[int]] = None,
    max_retries: Optional[int] = 5,
    initial_wait: Optional[float] = 10,
    backoff_factor: Optional[float] = 1,
) -> Callable:
    if retry_on_status_code is None:
        retry_on_status_code = set(range(400, 999))
    def should_retry(response: Response) -> bool:
        return response.status_code in retry_on_status_code

    def handle_retry_exhausted(response: Response, max_retries: int) -> None:
        raise HTTPStatusError(
            f"Failed after {max_retries} retries due to flaky network or rate limiting.",
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