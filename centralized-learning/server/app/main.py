import uvicorn

if __name__ == "__main__":
    # Tetap 0.0.0.0 agar bisa diakses antar container
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)