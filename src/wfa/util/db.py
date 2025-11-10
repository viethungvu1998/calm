import sqlite3 # Standard Python library for SQLite database interaction
import requests # Library for making HTTP requests (to download the SQL script)
from langchain_community.utilities.sql_database import SQLDatabase # LangChain utility to interact with SQL databases
from sqlalchemy import create_engine # SQLAlchemy function to create a database engine
from sqlalchemy.pool import StaticPool # SQLAlchemy connection pool class for in-memory databases

from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_engine():
    connection = sqlite3.connect(DATABASE_URL, check_same_thread=False)
    return create_engine(
        DATABASE_URL,
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )

def get_database():
    engine = get_engine()
    db = SQLDatabase(engine)
    return db
