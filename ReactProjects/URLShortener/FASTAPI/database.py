import sqlalchemy as db
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker


engine = db.create_engine("sqlite:///./shortener.db")
Base = declarative_base()
SessionLocal = sessionmaker(bind= engine)

