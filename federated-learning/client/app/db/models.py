from sqlalchemy import Column, Integer, String, DateTime, LargeBinary, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from .db import Base
from datetime import datetime

class UserLocal(Base):
    __tablename__ = "users_local"
    # SRS 2.1: Users Local
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    nrp = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    embeddings = relationship("Embedding", back_populates="user")
    attendance = relationship("AttendanceLocal", back_populates="user")

class Embedding(Base):
    __tablename__ = "embeddings"
    # SRS 2.2: Embeddings (Enkripsi Wajib)
    embedding_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users_local.user_id"))
    
    # Mengganti 'vector' biasa dengan komponen enkripsi
    encrypted_embedding = Column(LargeBinary) # Hasil AES-256
    iv = Column(LargeBinary)   # Initialization Vector
    salt = Column(LargeBinary) # Untuk derivasi key
    
    # Simpan gambar asli (terenkripsi) untuk Training Backbone
    encrypted_image = Column(LargeBinary, nullable=True) 
    
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("UserLocal", back_populates="embeddings")

class AttendanceLocal(Base):
    __tablename__ = "attendance_local"
    # SRS 2.3: Attendance Local
    log_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users_local.user_id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    confidence = Column(Float) # Skor kecocokan
    sent_to_server = Column(Boolean, default=False) # Status sinkronisasi

    user = relationship("UserLocal", back_populates="attendance")

class LocalConfig(Base):
    __tablename__ = "local_config"
    # SRS 2.4: Konfigurasi Lokal
    id = Column(Integer, primary_key=True)
    edge_id = Column(String)
    model_version = Column(Integer) # Versi model head saat ini
    last_sync = Column(DateTime)
    
    # Untuk logging performa training lokal (opsional tapi disarankan SRS)
    last_training_loss = Column(Float)
    last_training_accuracy = Column(Float)