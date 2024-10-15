import uuid
from typing import List, Optional
import psycopg
from psycopg import sql

from langchain_postgres import PostgresChatMessageHistory


def _my_create_table_and_index(table_name: str) -> List[sql.Composed]:
    """
    Make a SQL query to create a table.
    Our realization
    """
    index_name = f"idx_{table_name}_session_id"
    statements = [
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                session_id INT NOT NULL,
                message JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                add2knowledge_base BOOLEAN DEFAULT FALSE
            );
            """
        ).format(table_name=sql.Identifier(table_name)),
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} (session_id);
            """
        ).format(
            table_name=sql.Identifier(table_name), index_name=sql.Identifier(index_name)
        ),
    ]
    return statements


class MyPSQLChatMessageHistory(PostgresChatMessageHistory):
    def __init__(
            self,
            table_name: str,
            session_id: int,
            sync_connection: Optional[psycopg.Connection] = None,
            async_connection: Optional[psycopg.AsyncConnection] = None,
    ) -> None:
        super().__init__(table_name, str(uuid.uuid4()), sync_connection=sync_connection, async_connection=async_connection)
        self._session_id = session_id

    @staticmethod
    def create_tables(
            connection: psycopg.Connection,
            table_name: str,
            /,
    ) -> None:
        """Create the table schema in the database and create relevant indexes."""
        queries = _my_create_table_and_index(table_name)

        with connection.cursor() as cursor:
            for query in queries:
                cursor.execute(query)
        connection.commit()
