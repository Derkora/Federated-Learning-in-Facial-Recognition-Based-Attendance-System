from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, LargeBinary, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from .db import Base

class UserLocal(Base):
    __tablename__ = "users_local"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True) # ID unik dari server/global
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    embeddings = relationship("EmbeddingLocal", back_populates="user")
    attendances = relationship("AttendanceLocal", back_populates="user")

class EmbeddingLocal(Base):
    __tablename__ = "embeddings_local"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users_local.user_id"))
    embedding_data = Column(LargeBinary) 
    iv = Column(LargeBinary, nullable=True) 
    is_global = Column(Boolean, default=False) # True if synced from server
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("UserLocal", back_populates="embeddings")

class AttendanceLocal(Base):
    __tablename__ = "attendance_local"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users_local.user_id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float)
    device_id = Column(String)
    
    user = relationship("UserLocal", back_populates="attendances")
