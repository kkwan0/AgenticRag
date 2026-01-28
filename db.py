import psycopg2
from llama_index.vector_stores.postgres import PGVectorStore

from config import (
    DB_NAME, DB_HOST, DB_PORT, DB_USER, DB_PASSWORD,
    VECTOR_TABLE_NAME, EMBEDDING_DIM
)


def create_database(db_name: str = DB_NAME, drop_existing: bool = True) -> None:
    """Create a fresh database for vector storage."""
    conn = psycopg2.connect(
        dbname="postgres",
        host=DB_HOST,
        password=DB_PASSWORD,
        port=DB_PORT,
        user=DB_USER,
    )
    conn.autocommit = True

    with conn.cursor() as c:
        if drop_existing:
            c.execute(f"DROP DATABASE IF EXISTS {db_name}")
        c.execute(f"CREATE DATABASE {db_name}")

    conn.close()


def get_vector_store(
    db_name: str = DB_NAME,
    table_name: str = VECTOR_TABLE_NAME,
    embed_dim: int = EMBEDDING_DIM
) -> PGVectorStore:
    """Connect to and return the PGVectorStore."""
    return PGVectorStore.from_params(
        database=db_name,
        host=DB_HOST,
        password=DB_PASSWORD,
        port=DB_PORT,
        user=DB_USER,
        table_name=table_name,
        embed_dim=embed_dim,
    )