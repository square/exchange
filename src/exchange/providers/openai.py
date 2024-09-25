import os
import sys
from typing import Any, Dict, List, Tuple, Type
import json
import httpx

from exchange.message import Message
from exchange.providers.base import Provider, Usage
from exchange.providers.utils import (
    messages_to_openai_spec,
    openai_response_to_message,
    openai_single_message_context_length_exceeded,
    raise_for_status,
    tools_to_openai_spec,
)
from exchange.tool import Tool
from tenacity import retry, wait_fixed, stop_after_attempt
from exchange.providers.utils import retry_if_status
from exchange.content import Text

OPENAI_HOST = "https://api.openai.com/"

retry_procedure = retry(
    wait=wait_fixed(2),
    stop=stop_after_attempt(2),
    retry=retry_if_status(codes=[429], above=500),
    reraise=True,
)

USER_PROMPT_TEMPLATE = """
## Task

{system}

## Available Tools & Response Guidelines
You can either respond to the user with a message or compose tool calls. Your task is to translate user queries into appropriate tool calls or response messages in JSON format.

Follow these guidelines:
- Always respond with a valid JSON object containing the function to call and its parameters.
- Do not include any additional text or explanations.
- If you are responding with a message, include the key "message" with the response text.
- If you are composing a tool call, include the key "tool_calls" with a list of tool calls.

Here are some examples:

Example 1:
---
User Query: What's the weather like in New York today?
Available Functions:
1. get_current_weather(location)
2. get_forecast(location, days)

Response:
{"tool_calls": [{
  "function": "get_current_weather",
  "parameters": {
    "location": "New York"
  }
}]}
---

Example 2:
---
User Query: Find me Italian restaurants nearby.
Available Functions:
1. search_restaurants(cuisine, location)
2. get_restaurant_details(restaurant_id)

Response:
{"tool_calls": [{
  "function": "search_restaurants",
  "parameters": {
    "cuisine": "Italian",
    "location": "current location"
  }
}]}
---

Example 3:
---
User Query: Schedule a meeting with John tomorrow at 10 AM and show me the calendar.
Available Functions:
1. create_event(title, datetime, participants)
2. get_calendar()

Response:
{"tool_calls": [
    {
        "function": "create_event",
        "parameters": {
            "title": "Meeting with John",
            "datetime": "tomorrow at 10 AM",
            "participants": ["John"]
        }
    },
    {
        "function": "get_calendar",
        "parameters": {{}}
    }
]}
---

Example 4:
---
User Query: Hi there!
Available Functions:
1. create_event(title, datetime, participants)
2. get_calendar()

Response:
{
    "message": Hey! How can I help you today?
}

Now, given the following user query and available functions, respond with the appropriate function call in JSON format.

User Query: {user_query}
Available Functions:
{available_functions}

Response:
"""


def is_o1(model: str) -> bool:
    return model.startswith("o1")


def update_system_message(system: str, tools: Tuple[Tool]) -> str:
    if not tools:
        return system

    tool_names_str = ""
    for i, tool in enumerate(tools, start=1):
        tool_names_str += f"{i}. {tool.name}({', '.join(tool.parameters.keys())})\n"

    return USER_PROMPT_TEMPLATE.format(system=system, available_functions=tool_names_str)


def merge_consecutive_roles(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merges consecutive messages with the same role into a single message.

    Args:
        messages (List[Dict[str, Any]]): The list of messages to merge.

    Returns:
        List[Dict[str, Any]]: The list of messages with consecutive messages of the same role merged.
    """
    merged_messages = []
    current_role = None
    current_content = ""

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == current_role:
            current_content += "\n" + content
        else:
            if current_role:
                merged_messages.append({"role": current_role, "content": current_content.strip()})
            current_role = role
            current_content = content

    if current_role:
        merged_messages.append({"role": current_role, "content": current_content.strip()})

    return merged_messages


def convert_messages(original_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts original messages with 'system', 'user', 'assistant', and 'tool' roles
    into messages containing only 'user' and 'assistant' roles.

    Args:
        original_messages (List[Dict[str, Any]]): The original list of messages.

    Returns:
        List[Dict[str, Any]]: The converted list of messages with only 'user' and 'assistant' roles.
    """
    converted_messages = []
    tool_call_map = {}  # Maps tool_call_id to function details

    for msg in original_messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            # Convert 'system' messages to 'user' messages, optionally prefixing for clarity
            content = f"[System]: {content}"
            converted_messages.append({"role": "user", "content": content})

        elif role == "user":
            converted_messages.append({"role": "user", "content": content})

        elif role == "assistant":
            # Check for 'tool_calls' in the 'assistant' message
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tool_call in tool_calls:
                    tool_id = tool_call.get("id")
                    function = tool_call.get("function", {})
                    function_name = function.get("name")
                    arguments = function.get("arguments")

                    # Store the tool call details
                    tool_call_map[tool_id] = {"name": function_name, "arguments": arguments}

                # Optionally, you can indicate that the assistant initiated tool calls
                # For this implementation, we'll not modify the assistant's content
                # But this can be customized based on specific needs

            # Append the 'assistant' message
            assistant_content = msg.get("content", "")
            converted_messages.append({"role": "assistant", "content": assistant_content})

        elif role == "tool":
            # 'tool' messages are outputs from tool calls; convert them to 'assistant' messages
            # Find the corresponding tool call based on 'tool_call_id'
            tool_call_id = msg.get("tool_call_id")
            tool_output = msg.get("content", "")

            if tool_call_id and tool_call_id in tool_call_map:
                function_details = tool_call_map[tool_call_id]
                function_name = function_details["name"]

                # You can format the assistant's response to include tool output contextually
                # For simplicity, we'll append the tool output directly
                assistant_output = f"[Tool Output - {function_name}]: {tool_output}"
                converted_messages.append({"role": "assistant", "content": assistant_output})
            else:
                # If 'tool_call_id' is missing or not found, append the tool output as-is
                assistant_output = f"[Tool Output]: {tool_output}"
                converted_messages.append({"role": "assistant", "content": assistant_output})

    merged_converted_messages = merge_consecutive_roles(messages=converted_messages)
    return merged_converted_messages


class OpenAiProvider(Provider):
    """Provides chat completions for models hosted directly by OpenAI"""

    def __init__(self, client: httpx.Client) -> None:
        super().__init__()
        self.client = client

    @classmethod
    def from_env(cls: Type["OpenAiProvider"]) -> "OpenAiProvider":
        url = os.environ.get("OPENAI_HOST", OPENAI_HOST)
        try:
            key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise RuntimeError(
                "Failed to get OPENAI_API_KEY from the environment, see https://platform.openai.com/docs/api-reference/api-keys"
            )
        client = httpx.Client(
            base_url=url,
            auth=("Bearer", key),
            timeout=httpx.Timeout(60 * 10),
        )
        return cls(client)

    @staticmethod
    def get_usage(data: dict) -> Usage:
        usage = data.pop("usage")
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def complete(
        self,
        model: str,
        system: str,
        messages: List[Message],
        tools: Tuple[Tool],
        **kwargs: Dict[str, Any],
    ) -> Tuple[Message, Usage]:
        if is_o1(model):
            system_with_tools = update_system_message(system, tools)
            converted = convert_messages(
                [{"role": "system", "content": system}] + messages_to_openai_spec(messages),
            )
            messages = [Message(role=m["role"], content=[Text(text=m["content"])]) for m in converted]
            payload = dict(
                messages=messages_to_openai_spec(messages),
                model=model,
                **kwargs,
            )
        else:
            payload = dict(
                messages=[{"role": "system", "content": system}] + messages_to_openai_spec(messages),
                model=model,
                tools=tools_to_openai_spec(tools) if tools else [],
                **kwargs,
            )
        payload = {k: v for k, v in payload.items() if v}
        response = self._post(payload)

        # Check for context_length_exceeded error for single, long input message
        if "error" in response and len(messages) == 1:
            openai_single_message_context_length_exceeded(response["error"])

        message = openai_response_to_message(response)
        usage = self.get_usage(response)
        return message, usage

    @retry_procedure
    def _post(self, payload: dict) -> dict:
        response = self.client.post("v1/chat/completions", json=payload)
        return raise_for_status(response).json()


if __name__ == "__main__":
    original_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi there!"},
        {"role": "user", "content": "can you help me?"},
        {"role": "assistant", "content": "Sure! What do you need assistance with?", "tool_calls": []},
        {"role": "user", "content": "I need to book a flight to Paris on December 25th."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tcall_1",
                    "type": "function",
                    "function": {"name": "book_flight", "arguments": '{"destination": "Paris", "date": "2023-12-25"}'},
                }
            ],
        },
        {
            "role": "tool",
            "content": "Your flight to Paris on December 25th has been booked.",
            "tool_call_id": "tcall_1",
        },
        {"role": "assistant", "content": "I've booked your flight to Paris on December 25th."},
    ]

    converted = convert_messages(original_messages)

    messages = [Message(role=m["role"], content=[Text(text=m["content"])]) for m in converted]
    print("Messages:")
    for msg in messages:
        print(msg)
