from copy import deepcopy
from typing import List
from attrs import define, field


@define
class Checkpoint:
    """Checkpoint that counts the tokens in messages between the start and end index"""

    start_index: int = field(default=0)  # inclusive
    end_index: int = field(default=0)  # inclusive
    token_count: int = field(default=0)

    def __copy__(self) -> "Checkpoint":
        """
        Returns a deep copy of the Checkpoint object.
        """
        return Checkpoint(start_index=self.start_index, end_index=self.end_index, token_count=self.token_count)

    def __deepcopy__(self, _) -> "Checkpoint":
        """
        Returns a deep copy of the Checkpoint object.
        """
        return Checkpoint(start_index=self.start_index, end_index=self.end_index, token_count=self.token_count)


@define
class CheckpointData:
    """Aggregates all information about checkpoints"""

    total_token_count: int = field(default=0)
    checkpoints: List[Checkpoint] = field(default=[])
    message_index_offset: int = field(default=0)

    def __copy__(self) -> "CheckpointData":
        """Returns a deep copy of the CheckpointData object."""
        return CheckpointData(
            total_token_count=self.total_token_count,
            checkpoints=deepcopy(self.checkpoints),
            message_index_offset=self.message_index_offset,
        )

    def __deepcopy__(self, memo: dict) -> "CheckpointData":
        """Returns a deep copy of the CheckpointData object."""
        return CheckpointData(
            total_token_count=self.total_token_count,
            checkpoints=deepcopy(self.checkpoints, memo),
            message_index_offset=self.message_index_offset,
        )

    @property
    def last_message_index(self) -> int:
        if not self.checkpoints:
            return -1  # we don't have enough information to know
        return self.checkpoints[-1].end_index - self.message_index_offset

    def reset(self) -> None:
        """Resets the checkpoint data to its initial state."""
        self.checkpoints = []
        self.message_index_offset = 0
        self.total_token_count = 0

    def pop(self, index: int = -1) -> Checkpoint:
        """Removes and returns the checkpoint at the given index."""
        popped_checkpoint = self.checkpoints.pop(index)
        self.total_token_count = self.total_token_count - popped_checkpoint.token_count
        return popped_checkpoint
