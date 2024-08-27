from attrs import define, field


@define
class Checkpoint:
    """Checkpoint that counts the tokens in messages between the start and end index"""

    start_index: int = field(default=-1)
    end_index: int = field(default=0)
    token_count: int = field(default=0)
    latest_generated_tokens: int = field(default=0)
