from llama_index.vector_stores.postgres import PGVectorStore
import psycopg2
from config import DB_NAME, DB_HOST, DB_PASSWORD, DB_PORT, DB_USER, TABLE_NAME, EMBED_DIM

db_name = DB_NAME
host = DB_HOST
password = DB_PASSWORD
port = DB_PORT
user = DB_USER

def create_database():
    # Create the Postgres database if it doesn't exist
    conn = psycopg2.connect(
        dbname="postgres",
        host=host,
        password=password,
        port=port,
        user=user,
    )
    conn.autocommit = True

    with conn.cursor() as c: # remove active connections to allow drop
        c.execute(f"""
                SELECT pg_terminate_backend(pid) 
                FROM pg_stat_activity 
                WHERE datname = '{db_name}' AND pid <> pg_backend_pid()
            """)
        c.execute(f"DROP DATABASE IF EXISTS {db_name}")
        c.execute(f"CREATE DATABASE {db_name}")

    conn.close()


def connect_vector_store() -> PGVectorStore: # Connect to the Postgres vector store
    return PGVectorStore.from_params(
        database=DB_NAME,
        host=DB_HOST,
        password=DB_PASSWORD,
        port=DB_PORT,
        user=DB_USER,
        table_name=TABLE_NAME,
        embed_dim=EMBED_DIM,
    )
