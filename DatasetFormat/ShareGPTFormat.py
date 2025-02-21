from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ConversationTurn:
    from_attribute: Optional[str]
    value: str = Optional[str]
@dataclass
class ShareGPTFormat:
    conversations: Optional[List[ConversationTurn]] = None
    system: any = None
    tools: any = None
