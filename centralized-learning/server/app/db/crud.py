from sqlalchemy.orm import Session
from . import models, schemas
from datetime import datetime, timedelta, timezone

def register_client(db: Session, data: schemas.ClientBase):
    client = models.Client(
        edge_id=data.edge_id,
        ip_address=data.ip_address,
        status=data.status,
        last_seen=datetime.now(timezone(timedelta(hours=7)))
    )
    db.add(client)
    db.commit()
    db.refresh(client)
    return client
