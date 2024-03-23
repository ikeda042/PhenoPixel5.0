from sqlalchemy import Column, Integer, String, BLOB, FLOAT
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Cell(Base):
    __tablename__ = 'cells'
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label  = Column(Integer)
    perimeter = Column(FLOAT)
    area = Column(FLOAT)
    img_ph = Column(BLOB) 
    img_fluo1 = Column(BLOB)
    img_fluo2 = Column(BLOB)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)