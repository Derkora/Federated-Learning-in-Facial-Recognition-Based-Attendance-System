import base64

@app.post(api_training_get_label)
async def get_label(data: dict, db: Session = Depends(get_db)):
    nrp = data.get("nrp")
    name = data.get("name", "Unknown")
    edge_id = data.get("client_id", "edge-1")
    emb_b64 = data.get("embedding")
    
    # Dekode embedding dari Base64
    emb_bytes = (
        base64.b64decode(emb_b64)
        if emb_b64 else None
    )
 
    # Periksa apakah NRP sudah terdaftar
    user = db.query(UserGlobal).filter_by(nrp=nrp).first()
    if not user:
        # Tambah entri baru (ID terbuat secara otomatis)
        user = UserGlobal(
            nrp=nrp,
            name=name,
            registered_edge_id=edge_id,
            embedding=emb_bytes
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        # Perbarui nama atau embedding
        user.name = name
        if emb_bytes:
            user.embedding = emb_bytes
        db.commit()
    
    # Masukkan NRP ke daftar data aktif
    if nrp not in fl_manager.received_data:
        fl_manager.received_data.append(nrp)
        
    return {"nrp": nrp, "label": user.user_id}
