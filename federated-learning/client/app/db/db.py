from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    """Buat semua tabel jika belum ada.
    Pastikan models sudah di-import di file pemanggil sebelum fungsi ini dipanggil.
    """
    Base.metadata.create_all(bind=engine)
    print("[DB] Tables initialized.")

# Dependency FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
