from database import Base
from sqlalchemy import Column, Integer, String

class URL(Base):
    __tablename__ = "urls"
    id = Column(Integer, autoincrement= True, primary_key= True)
    alias = Column(String, nullable=False, unique=True)
    original_url = Column(String, nullable=False)
    clicks = Column(Integer, default= 0) 