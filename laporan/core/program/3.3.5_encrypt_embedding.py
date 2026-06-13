import os
import pickle
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

def encrypt_embedding(self, embedding_numpy):
    # Serialisasi array numpy embedding wajah menjadi bytes
    data_bytes = pickle.dumps(embedding_numpy)
    
    # Buat Initialization Vector (IV) acak sepanjang 16 bytes
    iv = os.urandom(16)
        
    # Inisialisasi Cipher AES-256 dalam mode CBC
    cipher = Cipher(
        algorithms.AES(self.key), 
        modes.CBC(iv), 
        backend=default_backend()
    )
    encryptor = cipher.encryptor()
 
    # Lakukan padding data bytes menggunakan standar PKCS7
    padder = padding.PKCS7(128).padder()
    padded_data = (
        padder.update(data_bytes) + 
        padder.finalize()
    )
 
    # Proses enkripsi data hasil padding
    encrypted_data = (
        encryptor.update(padded_data) + 
        encryptor.finalize()
    )
 
    return encrypted_data, iv
