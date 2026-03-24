from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from datetime import datetime
from .db import Base

class FLSession(Base):
    __tablename__ = "fl_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    status = Column(String, default="active") # active, completed, failed
    
    rounds = relationship("FLRound", back_populates="session")

class FLRound(Base):
    __tablename__ = "fl_rounds"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("fl_sessions.session_id"))
    round_number = Column(Integer)
    loss = Column(Float)
    metrics = Column(String) # JSON string of metrics
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    session = relationship("FLSession", back_populates="rounds")

class GlobalModel(Base):
    __tablename__ = "global_models"
    
    id = Column(Integer, primary_key=True, index=True)
    version = Column(Integer, default=0)
    weights = Column(LargeBinary) # Serialized global backbone weights
    last_updated = Column(DateTime, default=datetime.utcnow)

class Client(Base):
    __tablename__ = "clients"
    edge_id = Column(String, primary_key=True) 
    name = Column(String)
    ip_address = Column(String)
    status = Column(String) # online/offline
    last_seen = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    attendance_recap = relationship("AttendanceRecap", back_populates="client")

class UserGlobal(Base):
    __tablename__ = "users_global"
    user_id = Column(Integer, primary_key=True, autoincrement=True) 
    name = Column(String)
    nrp = Column(String, unique=True)
    photo_path = Column(String, nullable=True) 
    embedding = Column(LargeBinary, nullable=True) 
    registered_edge_id = Column(String, ForeignKey("clients.edge_id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    attendance = relationship("AttendanceRecap", back_populates="user")

class AttendanceRecap(Base):
    __tablename__ = "attendance_recap"
    recap_id = Column(Integer, primary_key=True, index=True) 
    user_id = Column(Integer, ForeignKey("users_global.user_id"))
    edge_id = Column(String, ForeignKey("clients.edge_id")) 
    timestamp = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float) 
    lecture_id = Column(String, nullable=True)

    user = relationship("UserGlobal", back_populates="attendance")
    client = relationship("Client", back_populates="attendance_recap")
