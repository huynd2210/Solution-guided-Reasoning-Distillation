import sqlite3
from icecream import ic

from models.DatabaseSchema import DatabaseSchema, Column, Table

def retrieveDatabaseSchema(db_path: str, include_sample_data: bool = False, sample_size: int = 5) -> DatabaseSchema:
    """
    Retrieve schema information from a SQLite database using dataclasses.

    Args:
        db_path (str): Path to the SQLite database file
        include_sample_data (bool): If True, includes first row of data for each table

    Returns:
        DatabaseSchema object containing all schema information
    """
    global conn
    try:
        ic(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all tables in the database
        cursor.execute("""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        tables = cursor.fetchall()

        schema_tables = []

        for table in tables:
            table_name = table[0]

            # Get column information
            # cursor.execute(f"PRAGMA table_info({table_name})")
            cursor.execute(f'PRAGMA table_info("{table_name}")')

            columns_info = cursor.fetchall()

            # Get sample data if requested
            sample_data = None
            if include_sample_data:
                cursor.execute(f'SELECT * FROM "{table_name}" LIMIT {sample_size}')
                sample_data = cursor.fetchall()

            # Process column information
            columns = []
            for idx, col in enumerate(columns_info):
                sample_values = []
                for i in range(len(sample_data)):
                    sample_values.append(sample_data[i][idx]) if sample_data else None
                column = Column(
                    name=col[1],
                    type=col[2],
                    nullable=not col[3],  # notnull constraint
                    default=col[4],
                    primary_key=bool(col[5]),  # pk constraint
                    sample_value=sample_values if sample_values else None
                )
                columns.append(column)

            table_schema = Table(name=table_name, columns=columns)
            schema_tables.append(table_schema)

        return DatabaseSchema(tables=schema_tables)

    except sqlite3.Error as e:
        # raise Exception(f"Database error for {db_path}: str({e})")
        raise e
    finally:
        if 'conn' in locals():
            conn.close()

def formatSchemaForPrompt(schema: DatabaseSchema, simple=True, maxSampleLength=70) -> str:
    """
    Format DatabaseSchema in a YAML-like structure.

    Args:
        schema (DatabaseSchema): The database schema to format

    Returns:
        str: Formatted string representing the schema
    """
    lines = ["SCHEMA:"]

    for table in schema.tables:
        # Add table
        lines.append(f"- Table: {table.name}")

        # Add columns for this table
        for col in table.columns:
            # Format column with type
            if simple:
                lines.append(f"  - Column: {col.name}")
            else:
                lines.append(f"  - Column: {col.name} ({col.type})")
                appendColumnDescriptions(col, lines)

            # Add sample values if they exist
            if col.sample_value is not None:
                if isinstance(col.sample_value, (list, tuple)):
                    isTruncated = False
                    for i in range(len(col.sample_value)):
                        if len(str(col.sample_value[i])) > maxSampleLength:
                            col.sample_value[i] = str(col.sample_value[i])[:maxSampleLength] + "..."
                            isTruncated = True

                    samples = ", ".join(str(x) for x in col.sample_value)
                    if isTruncated:
                        samples += " (truncated)"

                    lines.append(f"    - Samples: [{samples}]")
                else:
                    lines.append(f"    - Sample: {col.sample_value}")

    return "\n".join(lines)


def appendColumnDescriptions(col, lines):
    # Add description based on constraints
    constraints = []
    if col.primary_key:
        constraints.append("Primary Key")
    if not col.nullable:
        constraints.append("NOT NULL")
    if col.default is not None:
        constraints.append(f"Default: {col.default}")
    if constraints:
        description = ", ".join(constraints)
        lines.append(f"    - Description: {description}")


def getDatabaseSchemaForPrompt(db_path: str, include_sample_data=True) -> str:
    """
    Get a formatted string representing the schema of a database for use in a prompt.

    Args:
        db_path (str): Path to the SQLite database file

    Returns:
        str: Formatted string representing the schema
    """
    schema = retrieveDatabaseSchema(db_path, include_sample_data)
    return formatSchemaForPrompt(schema)



