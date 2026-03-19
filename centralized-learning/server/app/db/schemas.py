from pydantic import BaseModel
from datetime import datetime

class ClientBase(BaseModel):
    id: str
    ip_address: str
    cl_status: str

class ClientResponse(ClientBase):
    last_seen: datetime
    class Config:
        orm_mode = True

class AttendanceRecapBase(BaseModel):
    user_id: str
    edge_id: str
    confidence: float
    lecture_id: str = None
