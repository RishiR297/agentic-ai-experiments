import sqlite3
import os

def get_db_connection():
    # Resolve the path to the db relative to this file
    base_dir = os.path.dirname(os.path.dirname(__file__))  # goes from utils/ to code/
    db_path = os.path.join(base_dir, "../db/output.db")     # goes up again to Use Case 1/db/
    db_path = os.path.abspath(db_path)  # make it absolute

    return sqlite3.connect(db_path)
