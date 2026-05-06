import os
import time
import json
from flask import Blueprint, request, jsonify, Response, send_file

# Inisialisasi Flask Blueprint untuk modul Video
video_server_bp = Blueprint('video_server', __name__)

STORED_VIDEOS_DIR = "stored_videos"
VIDEO_CACHES_DIR = "video_caches"
os.makedirs(STORED_VIDEOS_DIR, exist_ok=True)
os.makedirs(VIDEO_CACHES_DIR, exist_ok=True)

@video_server_bp.route('/video/upload', methods=['POST'])
def upload_video():
    """Mengunggah berkas video (.mp4) langsung ke server pusat (stored_videos)."""
    try:
        # Mengambil nama berkas dari Header request
        filename = request.headers.get("X-File-Name", f"video_{int(time.time())}.mp4")
        filename = "".join(c for c in filename if c.isalnum() or c in "._-").strip()
        if not filename.lower().endswith(".mp4"):
            filename += ".mp4"
            
        file_path = os.path.join(STORED_VIDEOS_DIR, filename)
        
        # Simpan chunk data secara langsung ke disk untuk menghemat memori
        with open(file_path, "wb") as f:
            f.write(request.data)
            
        return jsonify({"status": "success", "message": f"Video {filename} berhasil diunggah."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@video_server_bp.route('/video/list', methods=['GET'])
def list_videos():
    """Mengembalikan daftar nama berkas video (.mp4) yang tersimpan di server."""
    files = [f for f in os.listdir(STORED_VIDEOS_DIR) if f.lower().endswith(".mp4")]
    return jsonify({"status": "success", "videos": files})

@video_server_bp.route('/video/stream/<video_name>', methods=['GET'])
def stream_video(video_name):
    """
    Streaming video menggunakan Byte-Range Requests (HTTP 206) untuk mendukung 
    seeking/scrubbing durasi video di HTML5 Player secara real-time tanpa lag.
    """
    video_name = "".join(c for c in video_name if c.isalnum() or c in "._-").strip()
    file_path = os.path.join(STORED_VIDEOS_DIR, video_name)
    
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "Video tidak ditemukan"}), 404
        
    file_size = os.path.getsize(file_path)
    range_header = request.headers.get('Range', None)
    
    # Ukuran blok data maksimum yang akan dikirim (4 MB)
    chunk_size = 1024 * 1024 * 4
    start = 0
    end = file_size - 1
    
    if range_header:
        # Contoh format header: bytes=10000-20000
        try:
            byte_range = range_header.replace("bytes=", "").split("-")
            if byte_range[0]:
                start = int(byte_range[0])
            if len(byte_range) > 1 and byte_range[1]:
                end = int(byte_range[1])
            else:
                end = min(start + chunk_size, file_size - 1)
        except Exception:
            pass
            
    start = max(0, min(start, file_size - 1))
    end = max(start, min(end, file_size - 1))
    content_length = end - start + 1
    
    # Generator untuk membaca berkas secara efisien di RAM
    def generate_video_chunks():
        with open(file_path, "rb") as f:
            f.seek(start)
            bytes_to_send = content_length
            while bytes_to_send > 0:
                read_size = min(1024 * 64, bytes_to_send) # Read 64 KB per step
                data = f.read(read_size)
                if not data:
                    break
                bytes_to_send -= len(data)
                yield data
                
    response = Response(generate_video_chunks(), status=206, mimetype='video/mp4')
    response.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
    response.headers.add('Accept-Ranges', 'bytes')
    response.headers.add('Content-Length', str(content_length))
    return response

@video_server_bp.route('/video/cache/<video_name>', methods=['GET', 'POST'])
def handle_video_cache(video_name):
    """Mengambil atau menyimpan koordinat wajah (bounding boxes) yang sudah dideteksi client."""
    video_name = "".join(c for c in video_name if c.isalnum() or c in "._-").strip()
    cache_path = os.path.join(VIDEO_CACHES_DIR, f"{video_name}.json")
    
    if request.method == 'GET':
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                data = json.load(f)
            return jsonify({"status": "success", "cached": True, "data": data})
        return jsonify({"status": "success", "cached": False, "data": []})
        
    elif request.method == 'POST':
        try:
            cache_data = request.get_json(force=True)
            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=4)
            return jsonify({"status": "success", "message": "Koordinat deteksi wajah berhasil disimpan di cache server."})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
