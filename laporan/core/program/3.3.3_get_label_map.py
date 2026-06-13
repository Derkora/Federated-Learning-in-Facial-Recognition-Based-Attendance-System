@app.get(api_training_label_map)
async def get_label_map(db: Session = Depends(get_db)):
    # Ambil semua identitas terdaftar, urutkan berdasarkan NRP
    users = db.query(UserGlobal).order_by(UserGlobal.nrp).all()
    # Kembalikan array NRP terurut
    return [u.nrp for u in users]
