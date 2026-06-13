from typing import List, Dict, Any
from datetime import datetime
from fastapi import Body, Depends

@app.post("/api/attendance/sync")
async def sync_attendance(
    records: List[Dict[str, Any]] = Body(...), 
    db: Session = Depends(get_db)
):
    new_records = 0
    for rec in records:
        try:
            nrp = rec.get('user_id')
            user = (
                db.query(UserGlobal)
                .filter_by(nrp=nrp).first()
            )
            if not user:
                continue
            
            # Parsing timestamp ISO format
            ts_str = rec['timestamp']
            ts = datetime.fromisoformat(
                ts_str.replace('Z', '+00:00')
            )
                
            # Validasi duplikasi entry presensi
            exists = (
                db.query(AttendanceRecap)
                .filter_by(user_id=user.user_id, timestamp=ts)
                .first()
            )
            if not exists:
                conf = rec.get('confidence', 0.0)
                client_id = rec.get('client_id', 'unknown-edge')
                
                # Simpan rekap absensi global mahasiswa
                new_item = AttendanceRecap(
                    user_id=user.user_id, 
                    edge_id=client_id, 
                    timestamp=ts, 
                    confidence=conf
                )
                db.add(new_item)
                new_records += 1
        except Exception:
            continue
            
    db.commit()
    return {"status": "success", "new_records": new_records}
