from sqlalchemy import Column, String, Integer

from database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    uid = Column(String, unique=True, index=True)
    name = Column(String)
