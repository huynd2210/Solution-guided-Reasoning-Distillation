from dataclasses import dataclass
from typing import Optional


@dataclass
class SQLEvaluationEntry:
    """
    A dataclass to represent a SQL test case.
    """
    db_path: str
    generated_sql: str
    gold_sql: str
    question: str
    #Result is set after this entry is evaluated
    isCorrect: Optional[bool] = None
    response: Optional[str] = None