import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import pickle

class EmbeddingEncryptor:
    def __init__(self, key: bytes = None):
        # PERBAIKAN: Hapus 'b' sebelum string default value
        # os.getenv mengembalikan string, jadi defaultnya harus string juga agar bisa di-.encode()
        if key:
            self.key = key
        else:
            # Mengambil dari ENV (String) atau Default (String) -> lalu encode ke Bytes
            secret = os.getenv("AES_SECRET_KEY", "01234567890123456789012345678901")
            self.key = secret.encode()[:32]
            
        if len(self.key) != 32:
            raise ValueError("AES Key harus 32 bytes (256 bit)")

    def encrypt_embedding(self, embedding_numpy):
        """
        Input: numpy array
        Output: (encrypted_bytes, iv, salt)
        """
        # Serialisasi numpy ke bytes
        data_bytes = pickle.dumps(embedding_numpy)

        # Generate IV (Initialization Vector) acak
        iv = os.urandom(16)
        salt = os.urandom(16) 

        # Setup AES Cipher (CBC Mode)
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        # Padding (PKCS7) agar pas dengan blok AES
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data_bytes) + padder.finalize()

        # Encrypt
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

        return encrypted_data, iv, salt

    def decrypt_embedding(self, encrypted_data, iv):
        """
        Mengembalikan numpy array dari data terenkripsi
        """
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()

        padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

        unpadder = padding.PKCS7(128).unpadder()
        data_bytes = unpadder.update(padded_data) + unpadder.finalize()

        return pickle.loads(data_bytes)