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
    user_id = Column(String, nullable=True)


def migrate(dbname: str) -> None:
    engine = create_engine(f"sqlite:///backend/app/databases/{dbname}")
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        with engine.connect() as connection:
            inspector = engine.dialect.get_columns(connection, Cell.__tablename__)
            existing_columns = {col["name"] for col in inspector}

            # Rename img_fluo to img_fluo1 if it exists
            if "img_fluo" in existing_columns:
                try:
                    connection.execute(
                        text(
                            f"ALTER TABLE {Cell.__tablename__} RENAME COLUMN img_fluo TO img_fluo1"
                        )
                    )
                    print("Renamed column 'img_fluo' to 'img_fluo1'")
                except OperationalError as e:
                    print(f"Failed to rename column 'img_fluo': {e}")

            model_columns = {col.name for col in Cell.__table__.columns}
            missing_columns = model_columns - existing_columns

            if missing_columns:
                print(f"Missing columns in '{Cell.__tablename__}': {missing_columns}")
                for col_name in missing_columns:
                    col_type = next(
                        (
                            col.type
                            for col in Cell.__table__.columns
                            if col.name == col_name
                        ),
                        None,
                    )
                    if col_type:
                        alter_query = f"ALTER TABLE {Cell.__tablename__} ADD COLUMN {col_name} {col_type}"
                        try:
                            connection.execute(text(alter_query))
                            print(f"Added column '{col_name}' with type '{col_type}'")
                        except OperationalError as e:
                            print(f"Failed to add column '{col_name}': {e}")
                session.commit()
            else:
                print(f"No missing columns in '{Cell.__tablename__}'")
    except OperationalError as e:
        print(f"Error while migrating {dbname}: {e}")
    finally:
        session.close()


for i in os.listdir("backend/app/databases"):
    if i.endswith(".db"):
        print(i)
        migrate(i)
        print(f"migrated {i}")
