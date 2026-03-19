from sqlalchemy.orm import Session
from . import models, schemas
from datetime import datetime

def register_client(db: Session, data: schemas.ClientBase):
    client = models.Client(
        edge_id=data.id,
        ip_address=data.ip_address,
        status=data.cl_status,
        last_seen=datetime.utcnow()
    )
    db.add(client)
    db.commit()
    db.refresh(client)
    return client
