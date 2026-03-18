from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
import uvicorn

from app.db.db import engine, Base
from app.db import models
from app.utils.init import create_initial_head

Base.metadata.create_all(bind=engine)

# Import Controller API
from app.controllers import training_controller, client_controller, attendance_controller
from app.server_manager_instance import fl_manager

app = FastAPI(title="Federated Learning Server")

@app.on_event("startup")
def on_startup():
    print("[MAIN] Menjalankan pengecekan inisialisasi model...")
    create_initial_head()
    
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.include_router(training_controller.router, prefix="/api/training", tags=["training"])
app.include_router(client_controller.router, prefix="/api/clients", tags=["clients"])
app.include_router(attendance_controller.router, prefix="/api/attendance", tags=["attendance"])

@app.get("/")
async def dashboard(request: Request):
    # Ambil status langsung dari fl_manager
    current_status = fl_manager.status()
    
    # Render HTML dengan data status
    return templates.TemplateResponse("index.html", {
        "request": request,
        "status": current_status
    })

@app.get("/users")
async def user_management(request: Request):
    return templates.TemplateResponse("users.html", {
        "request": request
    })

@app.post("/action/start")
async def action_start_training():
    # Panggil logic start 
    fl_manager.start_training(rounds=10)
    # Redirect kembali ke halaman utama agar refresh
    return RedirectResponse(url="/", status_code=303)

@app.post("/action/stop")
async def action_stop_training():
    fl_manager.stop()
    return RedirectResponse(url="/", status_code=303)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)