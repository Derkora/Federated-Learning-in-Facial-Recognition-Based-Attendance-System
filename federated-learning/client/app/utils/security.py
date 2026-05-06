import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import pickle

class EmbeddingEncryptor:
    def __init__(self, key: bytes = None):
        if key:
            self.key = key
        else:
            # Ambil kunci rahasia AES dari environment variable (default 256-bit key)
            secret = os.getenv("AES_SECRET_KEY", "01234567890123456789012345678901")
            self.key = secret.encode()[:32]
            
        if len(self.key) != 32:
            raise ValueError("Kunci AES harus berukuran 32 bytes (256 bits)")

    # Enkripsi Embedding Wajah
    def encrypt_embedding(self, embedding_numpy):
        # Serialisasi array numpy embedding wajah menjadi bentuk bytes
        data_bytes = pickle.dumps(embedding_numpy)

        # Buat Initialization Vector (IV) acak sepanjang 16 bytes untuk mode CBC
        iv = os.urandom(16)
        
        # Inisialisasi Cipher AES-256 dalam mode CBC
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        # Lakukan padding data bytes menggunakan standar PKCS7 agar sesuai kelipatan block size
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data_bytes) + padder.finalize()

        # Proses enkripsi data hasil padding menjadi ciphertext terenkripsi
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

        return encrypted_data, iv

    # Dekripsi Embedding Wajah
    def decrypt_embedding(self, encrypted_data, iv):
        # Inisialisasi Cipher AES-256 dalam mode CBC untuk dekripsi
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()

        # Proses dekripsi ciphertext menjadi data beralgoritma padding
        padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

        # Hapus padding PKCS7 untuk mengembalikan data bytes asli
        unpadder = padding.PKCS7(128).unpadder()
        data_bytes = unpadder.update(padded_data) + unpadder.finalize()

        # Rekonstruksi (deserialisasi) bytes menjadi array numpy embedding wajah semula
        return pickle.loads(data_bytes)

encryptor = EmbeddingEncryptor()