import os
import requests
import time

class Client:
    def __init__(self):
        self.server_url = os.getenv("SERVER_URL", "http://localhost:8000")
        self.client_id = os.getenv("CLIENT_ID", "unknown_client")

    def lapor_diri(self):
        try:
            print(f"[{self.client_id}] Menghubungi pusat di {self.server_url}...")
            response = requests.get(f"{self.server_url}/ping", timeout=5)
            if response.status_code == 200:
                print(f"[{self.client_id}] Respon Pusat: {response.json()['message']}")
                return True
        except Exception as e:
            print(f"[{self.client_id}] Gagal konek: {e}")
            return False

    def run_forever(self):
        while True:
            if self.lapor_diri():
                print(f"[{self.client_id}] Koneksi Aman. Menunggu instruksi selanjutnya...")
                time.sleep(30) # Cek berkala tiap 30 detik
            else:
                print(f"[{self.client_id}] Reconnecting dalam 5 detik...")
                time.sleep(5)