from fastapi import FastAPI

class Server:
    def __init__(self):
        self.app = FastAPI(title="Centralized Attendance Server")
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/ping")
        async def health_check():
            return {"status": "online", "message": "Siap Komandan, Server Pusat Standby!"}

        @self.app.post("/register-client")
        async def register(client_id: str):
            return {"status": "success", "message": f"Client {client_id} terdata!"}

server_instance = Server()
app = server_instance.app