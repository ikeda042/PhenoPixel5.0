from sqlalchemy import create_engine, Column, Integer, String, BLOB, FLOAT, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError
import os

Base = declarative_base()


class Cell(Base):
    __tablename__ = "cells"
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label = Column(Integer)
    perimeter = Column(FLOAT)
    area = Column(FLOAT)
    img_ph = Column(BLOB)
    img_fluo1 = Column(BLOB, nullable=True)
    img_fluo2 = Column(BLOB, nullable=True)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)


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
        print(i)
        migrate(i)
        print(f"migrated {i}")
