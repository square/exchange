from unittest.mock import MagicMock

import pytest
from exchange.providers.retry_with_back_off_decorator import retry_httpx_request, retry_with_backoff
from httpx import HTTPStatusError, Response


def create_mock_function():
    mock_function = MagicMock()
    mock_function.side_effect = [3, 5, 7]
    return mock_function


def test_retry_with_backoff_retry_exhausted():
    mock_function = create_mock_function()
    handle_retry_exhausted_function = MagicMock()

    def should_try(result):
        return result < 7

    @retry_with_backoff(
        should_retry=should_try,
        max_retries=2,
        initial_wait=0,
        backoff_factor=0.001,
        handle_retry_exhausted=handle_retry_exhausted_function,
    )
    def test_func():
        return mock_function()

    assert test_func() == 5

    assert mock_function.call_count == 2
    handle_retry_exhausted_function.assert_called_once()
    handle_retry_exhausted_function.assert_called_with(5, 2)


def test_retry_with_backoff_retry_successful():
    mock_function = create_mock_function()
    handle_retry_exhausted_function = MagicMock()

    def should_try(result):
        return result < 4

    @retry_with_backoff(
        should_retry=should_try,
        max_retries=2,
        initial_wait=0,
        backoff_factor=0.001,
        handle_retry_exhausted=handle_retry_exhausted_function,
    )
    def test_func():
        return mock_function()

    assert test_func() == 5

    assert mock_function.call_count == 2
    handle_retry_exhausted_function.assert_not_called()


def test_retry_with_backoff_without_retry():
    mock_function = create_mock_function()
    handle_retry_exhausted_function = MagicMock()

    def should_try(result):
        return result < 2

    @retry_with_backoff(
        should_retry=should_try,
        max_retries=2,
        initial_wait=0,
        backoff_factor=0.001,
        handle_retry_exhausted=handle_retry_exhausted_function,
    )
    def test_func():
        return mock_function()

    assert test_func() == 3

    assert mock_function.call_count == 1
    handle_retry_exhausted_function.assert_not_called()


def create_mock_httpx_request_call_function(responses=[500, 429, 200]):
    mock_function = MagicMock()
    mock_responses = []
    for response_code in responses:
        response = MagicMock()
        response.status_code = response_code
        mock_responses.append(response)

    mock_function.side_effect = mock_responses
    return mock_function


def test_retry_httpx_request_backoff_retry_exhausted():
    mock_httpx_request_call_function = create_mock_httpx_request_call_function()

    @retry_httpx_request(retry_on_status_code=[500, 429], max_retries=2, initial_wait=0, backoff_factor=0.001)
    def test_func() -> Response:
        return mock_httpx_request_call_function()

    with pytest.raises(HTTPStatusError):
        test_func()

    assert mock_httpx_request_call_function.call_count == 2


def test_retry_httpx_request_backoff_retry_successful():
    mock_httpx_request_call_function = create_mock_httpx_request_call_function()

    @retry_httpx_request(retry_on_status_code=[500], max_retries=2, initial_wait=0, backoff_factor=0.001)
    def test_func() -> Response:
        return mock_httpx_request_call_function()

    assert test_func().status_code == 429

    assert mock_httpx_request_call_function.call_count == 2


def test_retry_httpx_request_backoff_without_retry():
    mock_httpx_request_call_function = create_mock_httpx_request_call_function()

    @retry_httpx_request(retry_on_status_code=[503], max_retries=2, initial_wait=0, backoff_factor=0.001)
    def test_func() -> Response:
        return mock_httpx_request_call_function()

    assert test_func().status_code == 500

    assert mock_httpx_request_call_function.call_count == 1


def test_retry_httpx_request_backoff_range():
    mock_httpx_request_call_function = create_mock_httpx_request_call_function(responses=[200])

    @retry_httpx_request(max_retries=2, initial_wait=0, backoff_factor=0.001)
    def test_func() -> Response:
        return mock_httpx_request_call_function()

    assert test_func().status_code == 200

    assert mock_httpx_request_call_function.call_count == 1


def test_retry_httpx_request_backoff_range_retry_never_succeed():
    mock_httpx_request_call_function = create_mock_httpx_request_call_function(responses=[401, 500, 500])

    @retry_httpx_request(max_retries=3, initial_wait=0, backoff_factor=0.001)
    def test_func() -> Response:
        return mock_httpx_request_call_function()

    # Never gets a successful response
    with pytest.raises(HTTPStatusError):
        f = test_func()
        # last error is 500
        assert f.status_code == 500

    # Has been retried 3 times
    assert mock_httpx_request_call_function.call_count == 3


def test_retry_httpx_request_backoff_range_retry_succeed():
    mock_httpx_request_call_function = create_mock_httpx_request_call_function(responses=[401, 500, 200])

    @retry_httpx_request(max_retries=3, initial_wait=0, backoff_factor=0.001)
    def test_func() -> Response:
        return mock_httpx_request_call_function()

    # Retries and raises no error
    f = test_func()
    assert f.status_code == 200

    # Has been retried 3 times
    assert mock_httpx_request_call_function.call_count == 3
