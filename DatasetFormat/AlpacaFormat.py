from dataclasses import dataclass

@dataclass
class AlpacaFormat:
    instruction: str = ""
    input: str = ""
    output: str = ""
