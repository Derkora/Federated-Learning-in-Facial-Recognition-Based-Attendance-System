import sys
import os

# Tambahkan path aplikasi ke sys.path
sys.path.append(os.getcwd())

from app.utils.logging import Logger

# Inisialisasi logger mandiri untuk skrip diagnostik
logger = Logger("/app/data/diagnostic.log", tag="DIAG")

try:
    from app.db.db import SessionLocal
    from app.db import models
    from sqlalchemy import func

    db = SessionLocal()
    try:
        # Cek jumlah baris di tabel TrainingRound
        count = db.query(models.TrainingRound).count()
        logger.info(f"Total baris di tabel training_rounds: {count}")

        if count > 0:
            # Tampilkan 5 data terakhir
            last_rounds = db.query(models.TrainingRound).order_by(models.TrainingRound.round_id.desc()).limit(5).all()
            logger.info("5 Ronde terakhir:")
            for r in last_rounds:
                logger.info(f"  - ID: {r.round_id}, No: {r.round_number}, Acc: {r.global_accuracy}, Loss: {r.global_loss}")
        else:
            logger.warn("Tabel training_rounds KOSONG.")
            
        # Cek juga tabel model_versions
        v_count = db.query(models.ModelVersion).count()
        logger.info(f"Total baris di tabel model_versions: {v_count}")

    except Exception as e:
        logger.error(f"Gagal query database: {e}")
    finally:
        db.close()
except Exception as e:
    logger.error(f"Gagal inisialisasi diagnostik: {e}")
