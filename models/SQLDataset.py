from dataclasses import dataclass

@dataclass
class SQLDataset:
    db_id: any = None
    query: any = None
    question: any = None