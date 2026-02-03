from dataclasses import dataclass


@dataclass
class ChatMessage:
    text: str = ""
    role: str = ""
    marker: str = ""
    expanded_text: str = ""
    expanded_role: str = ""
    tps: float = 0
