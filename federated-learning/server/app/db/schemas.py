from pydantic import BaseModel
from datetime import datetime

class ClientBase(BaseModel):
    id: str
    ip_address: str
    fl_status: str

class ClientResponse(ClientBase):
    last_seen: datetime
    class Config:
        orm_mode = True
