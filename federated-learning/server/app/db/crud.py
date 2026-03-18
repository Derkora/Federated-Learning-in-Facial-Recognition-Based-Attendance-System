from sqlalchemy.orm import Session
from . import models, schemas
from datetime import datetime

def register_client(db: Session, data: schemas.ClientBase):
    client = models.Client(
        id=data.id,
        ip_address=data.ip_address,
        fl_status=data.fl_status,
        last_seen=datetime.utcnow()
    )
    db.add(client)
    db.commit()
    db.refresh(client)
    return client
