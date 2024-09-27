import os
import re
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
from exchange.content import Text, ToolUse
from exchange.utils import create_object_id

OPENAI_HOST = "https://api.openai.com/"

retry_procedure = retry(
    wait=wait_fixed(2),
    stop=stop_after_attempt(2),
    retry=retry_if_status(codes=[429], above=500),
    reraise=True,
)

USER_PROMPT_TEMPLATE = """
## Instructions

{system}

## Available Tools & Response Guidelines
You can either respond to the user with a message or compose tool calls. Your task is to translate user queries into
appropriate tool calls or response messages in JSON format.

Follow these guidelines:
- Always respond with a valid JSON object containing the function to call and its parameters.
- Do not include any additional text or explanations.
- If you are responding with a message, include the key "message" with the response text.
- If you are composing a tool call, include the key "tool_calls" with a list of tool calls.

Here are some examples:

Example 1:
---
User Query:
What's the weather like in New York today?

Available Functions:
[
  {{
    "type": "function",
    "function": {{
      "name": "get_current_weather",
      "description": "Get the current weather for the specified location.",
      "parameters": {{
        "type": "object",
        "properties": {{
          "location": {{
            "type": "string",
            "description": "The location to get the weather for."
          }}
        }},
        "required": [
          "location"
        ]
      }}
    }}
  }},
  {{
    "type": "function",
    "function": {{
      "name": "get_forecast",
      "description": "Get the weather forecast for the specified location for the next 'days' days.",
      "parameters": {{
        "type": "object",
        "properties": {{
          "location": {{
            "type": "string",
            "description": "The location to get the weather forecast for."
          }},
          "days": {{
            "type": "integer",
            "description": "The number of days to get the forecast for."
          }}
        }},
        "required": [
          "location",
          "days"
        ]
      }}
    }}
  }}
]

Response:
```json
{{"tool_calls": [{{
  "function": "get_current_weather",
  "parameters": {{
    "location": "New York"
  }}
}}]}}
```
---

Example 2:
---
User Query:
Find me Italian restaurants nearby.

Available Functions:
[
  {{
    "type": "function",
    "function": {{
      "name": "search_restaurants",
      "description": "Search for restaurants of the specified cuisine near the given location.",
      "parameters": {{
        "type": "object",
        "properties": {{
          "cuisine": {{
            "type": "string",
            "description": "The type of cuisine to search for."
          }},
          "location": {{
            "type": "string",
            "description": "The location to search near."
          }}
        }},
        "required": [
          "cuisine",
          "location"
        ]
      }}
    }}
  }},
  {{
    "type": "function",
    "function": {{
      "name": "get_restaurant_details",
      "description": "Get the details for the specified restaurant.",
      "parameters": {{
        "type": "object",
        "properties": {{
          "restaurant_id": {{
            "type": "string",
            "description": "The unique identifier for the restaurant."
          }}
        }},
        "required": [
          "restaurant_id"
        ]
      }}
    }}
  }}
]

Response:
```json
{{"tool_calls": [{{
  "function": "search_restaurants",
  "parameters": {{
    "cuisine": "Italian",
    "location": "current location"
  }}
}}]}}
```
---

Example 3:
---
User Query:
Schedule a meeting with John tomorrow at 10 AM and show me the calendar.

Available Functions:
[
  {{
    "type": "function",
    "function": {{
      "name": "create_event",
      "description": "Create an event with the specified title, datetime, and participants.",
      "parameters": {{
        "type": "object",
        "properties": {{
          "title": {{
            "type": "string",
            "description": "The title of the event."
          }},
          "datetime": {{
            "type": "string",
            "description": "The date and time of the event."
          }},
          "participants": {{
            "type": "array",
            "items": {{
              "type": "string"
            }},
            "description": "The list of participants for the event."
          }}
        }},
        "required": [
          "title",
          "datetime",
          "participants"
        ]
      }}
    }}
  }},
  {{
    "type": "function",
    "function": {{
      "name": "get_calendar",
      "description": "Get the user's calendar.",
      "parameters": {{
        "type": "object",
        "properties": {{}},
        "required": []
      }}
    }}
  }}
]

Response:
```json
{{"tool_calls": [
    {{
        "function": "create_event",
        "parameters": {{
            "title": "Meeting with John",
            "datetime": "tomorrow at 10 AM",
            "participants": ["John"]
        }}
    }},
    {{
        "function": "get_calendar",
        "parameters": {{}}
    }}
]}}
```
---

Example 4:
---
User Query:
Hi there!

Available Functions:
[
  {{
    "type": "function",
    "function": {{
      "name": "create_event",
      "description": "Create an event with the specified title, datetime, and participants.",
      "parameters": {{
        "type": "object",
        "properties": {{
          "title": {{
            "type": "string",
            "description": "The title of the event."
          }},
          "datetime": {{
            "type": "string",
            "description": "The date and time of the event."
          }},
          "participants": {{
            "type": "array",
            "items": {{
              "type": "string"
            }},
            "description": "The list of participants for the event."
          }}
        }},
        "required": [
          "title",
          "datetime",
          "participants"
        ]
      }}
    }}
  }},
  {{
    "type": "function",
    "function": {{
      "name": "get_calendar",
      "description": "Get the user's calendar.",
      "parameters": {{
        "type": "object",
        "properties": {{}},
        "required": []
      }}
    }}
  }}
]

Response:
```json
{{
    "message": "Hey! How can I help you today?"
}}
```
---

Example 5:
---
User Query:
There is no user query. The last assistant message contained tool calls. Here are the tool results:
[{{"tool_use_id": "tool_use_a0ce4b0f4ff8476f99c77f43", "output":'"Your flight to London on 2024-03-14 has been booked.", "is_error": false}}]

Available Functions:
[
  {{
    "type": "function",
    "function": {{
      "name": "create_event",
      "description": "Create an event with the specified title, datetime, and participants.",
      "parameters": {{
        "type": "object",
        "properties": {{
          "title": {{
            "type": "string",
            "description": "The title of the event."
          }},
          "datetime": {{
            "type": "string",
            "description": "The date and time of the event."
          }},
          "participants": {{
            "type": "array",
            "items": {{
              "type": "string"
            }},
            "description": "The list of participants for the event."
          }}
        }},
        "required": [
          "title",
          "datetime",
          "participants"
        ]
      }}
    }}
  }},
  {{
    "type": "function",
    "function": {{
      "name": "get_calendar",
      "description": "Get the user's calendar.",
      "parameters": {{
        "type": "object",
        "properties": {{}},
        "required": []
      }}
    }}
  }}
]

Response:
```json
{{
    "message": "As requested, we have booked your flight to London on 2024-03-14. Please let me know if you need anything else."
}}
```

## Task

Now, given the following user query and available functions, respond with the appropriate function call in JSON format.

User Query:
{user_query}

Available Functions:
{available_functions}

Response:
""".strip()


def is_o1(model: str) -> bool:
    return model.startswith("o1")


def update_system_message(system: str, tools: Tuple[Tool], user_query: str) -> str:
    tool_descriptions = json.dumps(tools_to_openai_spec(tools), indent=2)

    return USER_PROMPT_TEMPLATE.format(system=system, available_functions=tool_descriptions, user_query=user_query)


def extract_code_blocks(text: str) -> list:
    # Regular expression to match code blocks
    code_blocks = re.findall(r"```(?:\w+\n)?(.*?)```", text, re.DOTALL)
    return code_blocks


def parse_o1_assistant_reply(reply_text: str) -> Message:
    error_msg = "ERROR: This is not a valid response. Please make sure the response is in JSON format and contains either a 'message' key or 'tool_calls' key."

    try:
        code_blocks = extract_code_blocks(reply_text)
        if code_blocks:
            # If code blocks are present, treat the first block as JSON response
            response_data = json.loads(code_blocks[0])
        else:
            response_data = json.loads(reply_text)
    except json.JSONDecodeError:
        # If parsing fails, treat the reply as regular text
        return Message(
            role="assistant",
            content=[Text(text=error_msg)],
        )

    content = []

    if "tool_calls" in response_data:
        tool_calls = response_data["tool_calls"]
        for tool_call in tool_calls:
            tool_use = ToolUse(
                id=create_object_id("tool_use"),
                name=tool_call["function"],
                parameters=tool_call["parameters"],
            )
            content.append(tool_use)
    elif "message" in response_data:
        message_text = response_data["message"]
        content.append(Text(text=message_text))
    else:
        # Unrecognized format, treat as regular text
        content.append(Text(text=error_msg))

    return Message(role="assistant", content=content)


def merge_consecutive_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not messages:
        return messages

    merged_messages = [messages[0]]

    for current_message in messages[1:]:
        last_message = merged_messages[-1]
        if current_message["role"] == last_message["role"]:
            # Merge contents
            last_message["content"] += "\n" + current_message["content"]
        else:
            merged_messages.append(current_message)

    return merged_messages


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
            base_url=url + "v1/",
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
            # Prepare the messages for o1 models
            # Find the last user message
            last_user_message_index = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].role == "user":
                    last_user_message_index = i
                    break

            if last_user_message_index is None:
                raise ValueError("No user message found in messages")

            last_user_message = messages[last_user_message_index]

            tool_results = last_user_message.tool_result
            user_query = last_user_message.text
            if tool_results:
                tool_result_str = "[\n" + "\n".join([json.dumps(tr.to_dict()) for tr in tool_results]) + "\n]"
                user_query = f"There is no user query. The last assistant message contained tool calls. Here are the tool results:\n{tool_result_str}"
            elif not user_query:
                user_query = "There is no user query. You can respond to the user with 'message'."

            # Update the system message (incorporated into the user message)
            combined_user_message_text = update_system_message(system, tools, user_query)

            # Prepare the messages to send
            messages_to_send = []

            # Process previous messages before last user message
            for message in messages[:last_user_message_index]:
                if message.role == "assistant":
                    messages_to_send.append(
                        {
                            "role": "assistant",
                            "content": message.text,
                        }
                    )
                elif message.role == "user":
                    # Include user messages (e.g., tool results)
                    if message.tool_result:
                        tool_result_texts = [tr.output for tr in message.tool_result]
                        tool_result_combined = "\n".join(tool_result_texts)
                        messages_to_send.append(
                            {
                                "role": "user",
                                "content": tool_result_combined,
                            }
                        )
                    else:
                        messages_to_send.append(
                            {
                                "role": "user",
                                "content": message.text,
                            }
                        )
                else:
                    pass  # Ignore other roles

            # Add the combined user message
            messages_to_send.append(
                {
                    "role": "user",
                    "content": combined_user_message_text,
                }
            )

            # Merge consecutive messages with the same role
            messages_to_send = merge_consecutive_messages(messages_to_send)

            payload = dict(
                messages=messages_to_send,
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

        if is_o1(model):
            assistant_reply = response["choices"][0]["message"]["content"]
            message = parse_o1_assistant_reply(assistant_reply)
        else:
            message = openai_response_to_message(response)

        usage = self.get_usage(response)
        return message, usage

    @retry_procedure
    def _post(self, payload: dict) -> dict:
        # Note: While OpenAI and Ollama mount the API under "v1", this is
        # conventional and not a strict requirement. For example, Azure OpenAI
        # mounts the API under the deployment name, and "v1" is not in the URL.
        # See https://github.com/openai/openai-openapi/blob/master/openapi.yaml
        response = self.client.post("chat/completions", json=payload)
        return raise_for_status(response).json()


if __name__ == "__main__":
    from exchange import Exchange, Text
    from exchange.moderators.passive import PassiveModerator
    import pprint

    def book_flight(destination: str, date: str):
        """Book a flight to the specified destination on the given date.

        Args:
            destination (str): The airport code for destination of the flight. E.g., "LAX" for Los Angeles.
            date (str): The date of the flight in "YYYY-MM-DD" format. E.g., "2023-12-25".
        """
        return f"Your flight to {destination} on {date} has been booked."

    system = "You are a helpful assistant"
    tools = [Tool.from_function(book_flight)]
    provider = OpenAiProvider.from_env()

    ex = Exchange(
        provider=provider,
        model="o1-mini",  # Use the 'o1' model
        system="You are a helpful assistant.",
        tools=tools,
        moderator=PassiveModerator(),
    )

    ex.add(Message(role="user", content=[Text(text="Hi there!"), Text(text="can you help me?")]))
    pprint.pp(ex)
    print("-" * 80)

    pprint.pp(ex.reply())
    print("-" * 80)

    ex.add(Message.user("I need to book a flight to Paris on December 25th."))
    pprint.pp(ex.reply())
    print("-" * 80)
