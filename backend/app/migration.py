from sqlalchemy import create_engine, Column, Integer, String, BLOB, FLOAT, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError
import os
from database import Base


def migrate(dbname: str) -> None:
    engine = create_engine(f"sqlite:///backend/app/databases/{dbname}")
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        for table_name in Base.metadata.tables.keys():
            result = session.execute(
                text(f"PRAGMA table_info({table_name})")
            ).fetchall()
            print(f"Table {table_name} columns:")
            for column_info in result:
                print(f" - {column_info[1]} (Type: {column_info[2]})")
    except OperationalError as e:
        print(f"Error while migrating {dbname}: {e}")
    finally:
        session.close()


for i in os.listdir("backend/app/databases"):
    if i.endswith(".db"):
        migrate(i)
        print(f"migrated {i}")
