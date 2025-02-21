from pydantic import BaseModel


class SQLQuery(BaseModel):
    # reasoning: str
    sql_query: str

