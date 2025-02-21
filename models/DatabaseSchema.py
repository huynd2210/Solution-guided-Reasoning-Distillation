from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import pandas as pd


@dataclass
class Column:
    name: str
    type: str
    nullable: bool
    primary_key: bool
    default: Optional[Any] = None
    sample_value: Optional[Any] = None


@dataclass
class Table:
    name: str
    columns: List[Column]


@dataclass
class DatabaseSchema:
    tables: List[Table]

    def to_dict(self) -> Dict[str, List[Dict]]:
        """Convert schema to dictionary format"""
        return {
            table.name: [
                {
                    'name': col.name,
                    'type': col.type,
                    'nullable': col.nullable,
                    'primary_key': col.primary_key,
                    'default': col.default,
                    **({'sample_value': col.sample_value} if col.sample_value is not None else {})
                }
                for col in table.columns
            ]
            for table in self.tables
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert schema to pandas DataFrame"""
        rows = []
        for table in self.tables:
            for col in table.columns:
                rows.append({
                    'table': table.name,
                    'column': col.name,
                    'type': col.type,
                    'nullable': col.nullable,
                    'primary_key': col.primary_key,
                    'default': col.default,
                    'sample_value': col.sample_value
                })
        return pd.DataFrame(rows)

