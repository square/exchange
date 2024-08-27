import json
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Tuple

from attrs import define, evolve, field
from tiktoken import get_encoding

from exchange.checkpoint import Checkpoint
from exchange.content import ToolResult, ToolUse
from exchange.message import Message
from exchange.moderators import ContextTruncate, Moderator
from exchange.providers import Provider, Usage
from exchange.tool import Tool


def validate_tool_output(output: str) -> None:
    """Validate tool output for the given model"""
    max_output_chars = 2**20
    max_output_tokens = 16000
    encoder = get_encoding("cl100k_base")
    if len(output) > max_output_chars or len(encoder.encode(output)) > max_output_tokens:
        raise ValueError("This tool call created an output that was too long to handle!")


@define(frozen=True)
class Exchange:
    """An exchange of messages with an LLM

    The exchange class is meant to be largely immutable, with only the message list
    growing once constructed. Use .replace to alter the model, tools, etc.

    The exchange supports tool usage, calling tools and letting the model respond when
    using the .reply method. It handles most forms of errors and sends those errors back
    to the model, to let it attempt to recover.
    """

    provider: Provider
    model: str
    system: str
    moderator: Moderator = field(default=ContextTruncate())
    tools: Tuple[Tool] = field(factory=tuple, converter=tuple)
    messages: List[Message] = field(factory=list)
    checkpoints: List[Checkpoint] = field(factory=list)

    @property
    def _toolmap(self) -> Mapping[str, Tool]:
        return {tool.name: tool for tool in self.tools}

    def replace(self, **kwargs: Dict[str, Any]) -> "Exchange":
        """Make a copy of the exchange, replacing any passed arguments"""
        if kwargs.get("messages") is None:
            kwargs["messages"] = deepcopy(self.messages)
        if kwargs.get("checkpoints") is None:
            kwargs["checkpoints"] = deepcopy(self.checkpoints)
        return evolve(self, **kwargs)

    def add(self, message: Message) -> None:
        """Add a message to the history."""
        if self.messages and message.role == self.messages[-1].role:
            raise ValueError("Messages in the exchange must alternate between user and assistant")
        self.messages.append(message)

    def generate(self) -> Message:
        """Generate the next message."""
        self.moderator.rewrite(self)
        try:
            message, usage = self.provider.complete(
                self.model,
                self.system,
                messages=self.messages,
                tools=self.tools,
            )
        except HTTPRetryFailedError as e:
            # pop off the last message

        self.add(message)
        # this has to come after adding the response
        self.add_checkpoint(usage)

        return message

    def reply(self, max_tool_use: int = 128) -> Message:
        """Get the reply from the underlying model.

        This will process any requests for tool calls, calling them immediately, and
        storing the intermediate tool messages in the queue. It will return after the
        first response that does not request a tool use

        Args:
            max_tool_use: The maximum number of tool calls to make before returning. Defaults to 128.
        """
        if max_tool_use <= 0:
            raise ValueError("max_tool_use must be greater than 0")
        response = self.generate()
        curr_iter = 1  # generate() already called once
        while response.tool_use:
            content = []
            for tool_use in response.tool_use:
                tool_result = self.call_function(tool_use)
                content.append(tool_result)
            self.add(Message(role="user", content=content))

            # We've reached the limit of tool calls - break out of the loop
            if curr_iter >= max_tool_use:
                # At this point, the most recent message is `Message(role='user', content=ToolResult(...))`
                response = Message.assistant(
                    f"We've stopped executing additional tool cause because we reached the limit of {max_tool_use}",
                )
                self.add(response)
                break
            else:
                response = self.generate()
                curr_iter += 1

        return response

    def call_function(self, tool_use: ToolUse) -> ToolResult:
        """Call the function indicated by the tool use"""
        tool = self._toolmap.get(tool_use.name)

        if tool is None or tool_use.is_error:
            output = f"ERROR: Failed to use tool {tool_use.id}.\nDo NOT use the same tool name and parameters again - that will lead to the same error."  # noqa: E501

            if tool_use.is_error:
                output += f"\n{tool_use.error_message}"
            elif tool is None:
                valid_tool_names = ", ".join(self._toolmap.keys())
                output += f"\nNo tool exists with the name '{tool_use.name}'. Valid tool names are: {valid_tool_names}"

            return ToolResult(tool_use_id=tool_use.id, output=output, is_error=True)

        try:
            if isinstance(tool_use.parameters, dict):
                output = json.dumps(tool.function(**tool_use.parameters))
            elif isinstance(tool_use.parameters, list):
                output = json.dumps(tool.function(*tool_use.parameters))
            else:
                raise ValueError(
                    f"The provided tool parameters, {tool_use.parameters} could not be interpreted as a mapping of arguments."  # noqa: E501
                )

            validate_tool_output(output)

            is_error = False
        except Exception as e:
            tb = traceback.format_exc()
            output = str(tb) + "\n" + str(e)
            is_error = True

        return ToolResult(tool_use_id=tool_use.id, output=output, is_error=is_error)

    def add_tool_use(self, tool_use: ToolUse) -> None:
        """Manually add a tool use and corresponding result

        This will call the implied function and add an assistant
        message requesting the ToolUse and a user message with the ToolResult
        """
        tool_result = self.call_function(tool_use)
        self.add(Message(role="assistant", content=[tool_use]))
        self.add(Message(role="user", content=[tool_result]))

    def add_checkpoint(self, usage: Usage) -> None:
        self.checkpoints.append(
            Checkpoint(
                start_index=(0 if not self.checkpoints else self.checkpoints[-1].end_index),
                end_index=len(self.messages),
                token_count=(
                    usage.total_tokens
                    if not self.checkpoints
                    else usage.total_tokens - sum(cp.token_count for cp in self.checkpoints)
                ),
                latest_generated_tokens=usage.output_tokens,
            )
        )
