from sqlalchemy import Column, Integer, String, PickleType, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    embeddings = relationship("FaceEmbedding", back_populates="user")

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    embedding = Column(PickleType) # Numpy array disimpan sebagai binary pickle
    user = relationship("User", back_populates="embeddings")