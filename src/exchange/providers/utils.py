import base64
import json
import re
from typing import Any, Dict, List, Tuple

import httpx
from exchange.content import Text, ToolResult, ToolUse
from exchange.message import Message
from exchange.tool import Tool


def raise_for_status(response: httpx.Response) -> httpx.Response:
    """Raise with reason text."""
    try:
        response.raise_for_status()
        return response
    except httpx.HTTPStatusError as e:
        response.read()
        if response.text:
            raise httpx.HTTPStatusError(f"{e}\n{response.text}", request=e.request, response=e.response)
        else:
            raise e


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def messages_to_openai_spec(messages: List[Message]) -> List[Dict[str, Any]]:
    messages_spec = []
    for message in messages:
        converted = {"role": message.role}
        output = []
        for content in message.content:
            print("message content", content)
            if isinstance(content, Text):
                converted["content"] = content.text
            elif isinstance(content, ToolUse):
                sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", content.name)
                converted.setdefault("tool_calls", []).append(
                    {
                        "id": content.id,
                        "type": "function",
                        "function": {
                            "name": sanitized_name,
                            "arguments": json.dumps(content.parameters),
                        },
                    }
                )
            elif isinstance(content, ToolResult):
                if content.output.startswith('"image:'):
                    image_path = content.output.replace('"image:', "").replace('"', "")
                    output.append(
                        {
                            "role": "tool",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "This tool result included an image that is uploaded in the next message.",
                                },
                            ],
                            "tool_call_id": content.tool_use_id,
                        }
                    )
                    # Note: it is possible to only do this when message == messages[-1]
                    # but it doesn't seem to hurt too much with tokens to keep this.
                    output.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"},
                                }
                            ],
                        }
                    )

                else:
                    output.append(
                        {
                            "role": "tool",
                            "content": content.output,
                            "tool_call_id": content.tool_use_id,
                        }
                    )

        if "content" in converted or "tool_calls" in converted:
            output = [converted] + output
        messages_spec.extend(output)
    return messages_spec


def tools_to_openai_spec(tools: Tuple[Tool]) -> Dict[str, Any]:
    tools_names = set()
    result = []
    for tool in tools:
        if tool.name in tools_names:
            # we should never allow duplicate tools
            raise ValueError(f"Duplicate tool name: {tool.name}")
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
        )
        tools_names.add(tool.name)
    return result


def openai_response_to_message(response: dict) -> Message:
    original = response["choices"][0]["message"]
    content = []
    text = original.get("content")

    '''
    actual text here and then if there is a tool call it follows: 
[TOOL_CALL]
```json
[
    {
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": "Update the plan by overwriting all current tasks. This can be used to update the status of a task. This update will be shown to the user directly; you do not need to reiterate it.",
            "parameters": {
                "tasks": [
                    {
                        "description": "Create a file named 'a.txt' with content 'a'",
                        "status": "planned"
                    }
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file at the specified path with the provided content. This will create any directories if they do not exist. The content will fully overwrite the existing file.",
            "parameters": {
                "path": "a.txt",
                "content": "a"
            }
        }
    }
]
```    
    
    '''
    
    # Split text from before the tool call if it is there, and set "tool_calls" to the JSON after it if it is present
    tool_calls = None
    if text:
        tool_call_marker = '[TOOL_CALL]'
        if tool_call_marker in text:
            text_before, tool_call_json = text.split(tool_call_marker, 1)
            print("tool_call_json", tool_call_json)
            text = text_before.strip()
            tool_call_json = tool_call_json.strip().strip('```json').strip().strip('```')
            tool_calls = json.loads(tool_call_json)

    if text:
        content.append(Text(text=text))

    if tool_calls is None:
        tool_calls = original.get("tool_calls")

    if tool_calls:
        for tool_call in tool_calls:
            try:
                function_name = tool_call["function"]["name"]
                if not re.match(r"^[a-zA-Z0-9_-]+$", function_name):
                    content.append(
                        ToolUse(
                            id=tool_call["id"],
                            name=function_name,
                            parameters=tool_call["function"]["arguments"],
                            is_error=True,
                            error_message=f"The provided function name '{function_name}' had invalid characters, it must match this regex [a-zA-Z0-9_-]+",  # noqa: E501
                        )
                    )
                else:
                    content.append(
                        ToolUse(
                            id=tool_call["id"],
                            name=function_name,
                            parameters=json.loads(tool_call["function"]["arguments"]),
                        )
                    )
            except json.JSONDecodeError:
                content.append(
                    ToolUse(
                        id=tool_call["id"],
                        name=tool_call["function"]["name"],
                        parameters=tool_call["function"]["arguments"],
                        is_error=True,
                        error_message=f"Could not interpret tool use parameters for id {tool_call['id']}: {tool_call['function']['arguments']}",  # noqa: E501
                    )
                )

    if text:
        content.append(Text(text=text))

    tool_calls = original.get("tool_calls")
    if tool_calls:
        for tool_call in tool_calls:
            try:
                function_name = tool_call["function"]["name"]
                # We occasionally see the model generate an invalid function name
                # sending this back to openai raises a validation error
                if not re.match(r"^[a-zA-Z0-9_-]+$", function_name):
                    content.append(
                        ToolUse(
                            id=tool_call["id"],
                            name=function_name,
                            parameters=tool_call["function"]["arguments"],
                            is_error=True,
                            error_message=f"The provided function name '{function_name}' had invalid characters, it must match this regex [a-zA-Z0-9_-]+",  # noqa: E501
                        )
                    )
                else:
                    content.append(
                        ToolUse(
                            id=tool_call["id"],
                            name=function_name,
                            parameters=json.loads(tool_call["function"]["arguments"]),
                        )
                    )
            except json.JSONDecodeError:
                content.append(
                    ToolUse(
                        id=tool_call["id"],
                        name=tool_call["function"]["name"],
                        parameters=tool_call["function"]["arguments"],
                        is_error=True,
                        error_message=f"Could not interpret tool use parameters for id {tool_call['id']}: {tool_call['function']['arguments']}",  # noqa: E501
                    )
                )

    return Message(role="assistant", content=content)


def openai_single_message_context_length_exceeded(error_dict: dict) -> None:
    code = error_dict.get("code")
    if code == "context_length_exceeded" or code == "string_above_max_length":
        raise InitialMessageTooLargeError(f"Input message too long. Message: {error_dict.get('message')}")


class InitialMessageTooLargeError(Exception):
    """Custom error raised when the first input message in an exchange is too large."""

    pass
