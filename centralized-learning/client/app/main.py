import os
import base64
import io
import time
import uvicorn
import cv2
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import requests
from collections import deque
from PIL import Image
from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks, Body
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.client_manager_instance import cl_client
from app.utils.preprocessing import image_processor

import tempfile

# Konfigurasi redistribusi temporary folder Starlette ke media eksternal (USB flashdrive symlink)
temp_dir = "app/static/temp"
os.makedirs(temp_dir, exist_ok=True)
tempfile.tempdir = temp_dir

# Konfigurasi endpoint API
# Endpoint Lokal Client
api_inference = "/api/inference"
api_results_latest = "/api/results/latest"
api_camera_toggle = "/api/camera/toggle"
api_logs = "/api/logs"
api_request_data = "/api/request-data"

# Endpoint Pengujian Video Lokal Client
api_test_video_prefix = "/api/test-video"
api_test_video_upload = f"{api_test_video_prefix}/upload"
api_test_video_local = f"{api_test_video_prefix}/local-path"
api_test_video_delete = f"{api_test_video_prefix}/delete"
api_test_video_metadata = f"{api_test_video_prefix}/metadata"
api_test_video_stream = f"{api_test_video_prefix}/stream"
api_test_video_events = f"{api_test_video_prefix}/events"

# Endpoint Proksi Server (Video)
api_video_prefix = "/api/video"
api_video_list = f"{api_video_prefix}/list"
api_video_cache = f"{api_video_prefix}/cache/{{video_name}}"
api_video_stream = f"{api_video_prefix}/stream/{{video_name}}"
api_video_delete = f"{api_video_prefix}/delete/{{video_name}}"
api_video_process = f"{api_video_prefix}/process-and-cache/{{video_name}}"

# Konfigurasi Aplikasi Client Terpusat (Centralized)
app = FastAPI(title="Centralized Edge Client")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Halaman Utama Presensi
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("attendance.html", {
        "request": request, 
        "client_id": cl_client.client_id,
        "model_version": cl_client.current_model_version
    })

# API Pengenalan Wajah (Inference)
@app.post(api_inference)
async def api_inference(data: dict):
    # Memastikan model dan aset sudah terunduh dari server
    if not cl_client.has_assets:
        return JSONResponse({"matched": "Unknown", "confidence": 0, "message": "Model belum siap"})
    
    start = time.time()
    try:
        img_data = base64.b64decode(data['image'])
        img_pil = Image.open(io.BytesIO(img_data)).convert('RGB')
        
        # Set as latest frame for MJPEG monitoring
        cl_client.latest_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Proses pengenalan wajah melalui controller attendance
        matched, confidence, _ = cl_client.attendance.process_inference(
            img_pil, cl_client.model, cl_client.reference_embeddings
        )
        
        latency = int((time.time() - start) * 1000)
        res = {
            "matched": matched, 
            "confidence": confidence, 
            "latency_ms": latency,
            "model_version": cl_client.current_model_version,
            "is_virtual": True # Browser frame is virtual
        }
        cl_client.latest_result = res
        return res
    except Exception as e:
        cl_client.logger.error(f"Gagal memproses inferensi: {e}")
        return JSONResponse({"matched": "Error", "error": str(e)}, status_code=400)

# Endpoint Monitoring Video (MJPEG Stream)
@app.get("/video_feed")
async def video_feed():
    def gen_frames():
        while True:
            if cl_client.latest_frame is not None:
                # Kompres ke JPEG dengan kualitas rendah (Optimize bandwidth)
                ret, buffer = cv2.imencode('.jpg', cl_client.latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.3) # Throttle FPS -> ~3 FPS

    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

# API Hasil Terakhir (untuk Polling UI Remote)
@app.get(api_results_latest)
async def get_latest_result():
    return cl_client.latest_result

@app.post(api_camera_toggle)
async def toggle_camera():
    is_on = cl_client.toggle_camera()
    return {"status": "success", "is_on": is_on}

@app.get(api_logs)
async def get_logs():
    """Mengambil log dari memori logger global."""
    try:
        logs = cl_client.logger.get_logs()
        return {"logs": "\n".join(logs)}
    except Exception as e:
        return {"logs": f"Error membaca log: {str(e)}"}

# API Permintaan Data dari Server
@app.post(api_request_data)
async def request_data():
    cl_client.logger.info("Server meminta unggah dataset.")
    success, msg = cl_client.management.package_and_upload()
    return {"status": "success" if success else "error", "message": msg}

class LimitedList(list):
    def append(self, item):
        super().append(item)
        if len(self) > 1000:
            self.pop(0)

# Endpoints pengujian file video
video_detection_events = LimitedList()
video_source_path = "app/static/temp/test_input.mp4"

# Proksi endpoint ke server-side streaming dan cache
@app.get(api_video_list)
async def proxy_list_videos():
    try:
        res = requests.get(f"{cl_client.server_url}{api_video_list}", timeout=5)
        return JSONResponse(res.json(), status_code=res.status_code)
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Server offline atau tidak terjangkau: {e}"}, status_code=500)

@app.get(api_video_cache)
async def proxy_get_video_cache(video_name: str):
    try:
        res = requests.get(f"{cl_client.server_url}{api_video_cache.format(video_name=video_name)}", timeout=5)
        return JSONResponse(res.json(), status_code=res.status_code)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get(api_video_stream)
def proxy_video_stream(video_name: str, request: Request):
    """Stream video from CL Server to browser with Byte-Range support."""
    headers = {}
    range_header = request.headers.get("range")
    if range_header:
        headers["range"] = range_header
    try:
        # Ambil data video dari server secara streaming
        server_res = requests.get(
            f"{cl_client.server_url}{api_video_stream.format(video_name=video_name)}",
            headers=headers,
            stream=True,
            timeout=10800
        )
        response_headers = {
            "content-type": server_res.headers.get("content-type", "video/mp4"),
            "accept-ranges": "bytes",
        }
        if server_res.headers.get("content-range"):
            response_headers["content-range"] = server_res.headers.get("content-range")
        if server_res.headers.get("content-length"):
            response_headers["content-length"] = server_res.headers.get("content-length")

        def chunk_generator():
            for chunk in server_res.iter_content(chunk_size=1024*64):
                yield chunk

        return StreamingResponse(
            chunk_generator(),
            status_code=server_res.status_code,
            headers=response_headers
        )
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Gagal melakukan streaming dari server: {e}"}, status_code=500)

@app.post(api_video_cache)
async def proxy_save_video_cache(video_name: str, data: list = Body(...)):
    try:
        res = requests.post(f"{cl_client.server_url}{api_video_cache.format(video_name=video_name)}", json=data, timeout=10800)
        return JSONResponse(res.json(), status_code=res.status_code)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.delete(api_video_cache)
async def proxy_delete_video_cache(video_name: str):
    try:
        res = requests.delete(f"{cl_client.server_url}{api_video_cache.format(video_name=video_name)}", timeout=5)
        return JSONResponse(res.json(), status_code=res.status_code)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.delete(api_video_delete)
async def proxy_delete_video(video_name: str):
    try:
        res = requests.delete(f"{cl_client.server_url}{api_video_delete.format(video_name=video_name)}", timeout=5)
        return JSONResponse(res.json(), status_code=res.status_code)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post(api_video_process)
async def process_and_cache_video(video_name: str, background_tasks: BackgroundTasks):
    """
    Menjalankan pemrosesan AI di Edge secara asinkron (background) pada aliran video stream server,
    menggunakan Frame Skipping, lalu mengirimkan hasilnya ke server untuk di-cache.
    """
    # Hapus cache lama di server agar polling mendeteksi cache kosong dan tidak langsung lompat sukses
    try:
        requests.delete(f"{cl_client.server_url}{api_video_cache.format(video_name=video_name)}", timeout=5)
    except Exception as e:
        cl_client.logger.warning(f"Gagal menghapus cache lama di server: {e}")
        
    background_tasks.add_task(run_offline_detection_and_cache, video_name)
    return {"status": "success", "message": "Pemrosesan video dimulai di latar belakang perangkat edge."}

def run_offline_detection_and_cache(video_name: str):
    cl_client.logger.info(f"[EDGE PROCESS] Memulai pemrosesan model AI edge untuk video: {video_name}")
    
    stream_url = f"{cl_client.server_url}{api_video_stream.format(video_name=video_name)}"
    
    # Unduh berkas video secara lokal untuk menghindari batasan aliran http OpenCV di dalam container
    temp_local_path = os.path.join(cl_client.data_path, f"temp_process_{int(time.time())}_{video_name}")
    try:
        cl_client.logger.info(f"[EDGE PROCESS] Mengunduh video simulasi dari server: {stream_url}")
        res = requests.get(stream_url, stream=True, timeout=10800)
        res.raise_for_status()
        with open(temp_local_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        cl_client.logger.info(f"[EDGE PROCESS] Selesai mengunduh. Membuka video lokal dengan OpenCV...")
        cap = cv2.VideoCapture(temp_local_path)
    except Exception as download_err:
        cl_client.logger.error(f"Gagal mengunduh berkas video dari server: {download_err}")
        if os.path.exists(temp_local_path):
            try: os.remove(temp_local_path)
            except: pass
        return

    if not cap.isOpened():
        cl_client.logger.error(f"Gagal membuka berkas video sementara: {temp_local_path}")
        if os.path.exists(temp_local_path):
            try: os.remove(temp_local_path)
            except: pass
        return
        
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        
        DEVICE = torch.device("cpu")
        threshold = cl_client.threshold
        
        detection_cache = []
        frame_idx = 0
        SKIP_FRAMES = 5
        
        prediction_buffer = deque(maxlen=5)
        last_face_frame = -100
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if frame_idx % SKIP_FRAMES != 0:
                continue
                
            # Gunakan resolusi asli video untuk akurasi jarak jauh (2.5m - 3m)
            frame_resized = frame
            scale = 1.0
                
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            face_tensor, box, prob = image_processor.detect_face(img_pil)
            if face_tensor is not None and box is not None:
                x1, y1, x2, y2 = [int(b / scale) for b in box]
                
                if frame_idx - last_face_frame > 10:
                    prediction_buffer.clear()
                last_face_frame = frame_idx
                
                face_tensor_ready = image_processor.prepare_for_model(face_tensor)
                with torch.no_grad():
                    if cl_client.model is not None:
                        cl_client.model.eval()
                        emb_orig = cl_client.model(face_tensor_ready)
                        face_flipped = torch.flip(face_tensor_ready, dims=[3])
                        emb_mirror = cl_client.model(face_flipped)
                        query_emb_tensor = F.normalize((emb_orig + emb_mirror) / 2, p=2, dim=1).view(1, -1)
                        prediction_buffer.append(query_emb_tensor)
                        
                        mean_emb_tensor = torch.stack(list(prediction_buffer)).mean(0).view(1, -1)
                        mean_emb_tensor = F.normalize(mean_emb_tensor, p=2, dim=1)
                        
                        user_ids = list(cl_client.reference_embeddings.keys())
                        ref_list = []
                        for nrp in user_ids:
                            ref = cl_client.reference_embeddings[nrp]
                            ref = torch.tensor(ref).to(DEVICE).view(-1)
                            ref_list.append(ref)
                        
                        best_match = "Unknown"
                        max_sim = 0.0
                        best_candidate = "None"
                        
                        if ref_list:
                            ref_tensor = torch.stack(ref_list) # Shape: (num_users, 128)
                            ref_tensor = F.normalize(ref_tensor, p=2, dim=1)
                            similarities = torch.mm(mean_emb_tensor, ref_tensor.t()).cpu().numpy()[0]
                            max_idx = np.argmax(similarities)
                            max_sim = float(similarities[max_idx])
                            best_candidate = user_ids[max_idx]
                            if max_sim >= threshold:
                                best_match = best_candidate
                                
                        cand_nrp = best_candidate.split("_")[0] if "_" in best_candidate else best_candidate
                        
                        if max_sim >= threshold:
                            label = best_match.split("_")[0] if "_" in best_match else best_match
                            cl_client.logger.success(f"[EDGE PROCESS] Frame {frame_idx}: Berhasil mengidentifikasi {label} (Sim: {max_sim:.4f})")
                        else:
                            if best_candidate != "None":
                                label = f"Unknown ({cand_nrp})"
                                cl_client.logger.info(f"[EDGE PROCESS] Frame {frame_idx}: Terdeteksi mirip {cand_nrp} (Sim: {max_sim:.4f}), di bawah threshold {threshold:.2f} (Ditandai sebagai {label})")
                            else:
                                label = "Unknown"
                                cl_client.logger.info(f"[EDGE PROCESS] Frame {frame_idx}: Tidak ada kemiripan terdeteksi.")
                            
                        detection_cache.append({
                            "frame": frame_idx,
                            "seconds": round(frame_idx / fps, 2),
                            "box": [x1, y1, x2, y2],
                            "label": label,
                            "confidence": float(max_sim)
                        })
        
        cap.release()
        cl_client.logger.info(f"[EDGE PROCESS] Selesai mendeteksi video {video_name}. Mengirim {len(detection_cache)} data ke cache server...")
        
        try:
            res = requests.post(f"{cl_client.server_url}{api_video_cache.format(video_name=video_name)}", json=detection_cache, timeout=10800)
            if res.status_code == 200:
                cl_client.logger.success(f"[EDGE PROCESS SUCCESS] Cache deteksi berhasil disimpan untuk {video_name}!")
            else:
                cl_client.logger.error(f"Gagal menyimpan cache ke server: {res.status_code}")
        except Exception as e:
            cl_client.logger.error(f"Gagal menghubungi server untuk mengirim cache: {e}")
            
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if os.path.exists(temp_local_path):
            try:
                os.remove(temp_local_path)
                cl_client.logger.info(f"[EDGE PROCESS] Berkas video sementara {temp_local_path} berhasil dihapus dari edge.")
            except Exception as rm_err:
                cl_client.logger.warning(f"Gagal menghapus berkas video sementara: {rm_err}")

@app.get("/test-video", response_class=HTMLResponse)
async def test_video_page(request: Request):
    return templates.TemplateResponse("test-video.html", {
        "request": request,
        "client_id": cl_client.client_id,
        "model_version": cl_client.current_model_version
    })

@app.post(api_test_video_upload)
async def upload_test_video(request: Request):
    global video_source_path
    try:
        temp_dir = "app/static/temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, "test_input.mp4")
        
        # Hapus file lama jika ada untuk menghemat ruang sebelum menulis yang baru
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Tulis secara streaming chunk-by-chunk langsung ke disk (mencegah cache /tmp raksasa)
        with open(temp_path, "wb") as buffer:
            async for chunk in request.stream():
                buffer.write(chunk)
            
        video_source_path = temp_path
        cl_client.logger.info("Video berhasil diunggah secara streaming.")
        return {"status": "success", "message": "Video uploaded successfully"}
    except Exception as e:
        cl_client.logger.error(f"Gagal mengunggah video: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post(api_test_video_local)
async def use_local_video_path(data: dict):
    global video_source_path
    path = data.get("path", "").strip()
    if not path or not os.path.exists(path):
        return JSONResponse({"status": "error", "message": "Path file lokal tidak valid atau tidak ditemukan"}, status_code=400)
    
    video_source_path = path
    cl_client.logger.info(f"Menggunakan path video lokal (Tanpa Upload): {path}")
    return {"status": "success", "message": "Using local video path"}

@app.post(api_test_video_delete)
async def delete_test_video():
    global video_source_path
    try:
        temp_path = "app/static/temp/test_input.mp4"
        if os.path.exists(temp_path):
            os.remove(temp_path)
            cl_client.logger.info("Video temporer berhasil dihapus dari penyimpanan.")
        
        video_source_path = temp_path
        return {"status": "success", "message": "Video temporary file deleted successfully"}
    except Exception as e:
        cl_client.logger.error(f"Gagal menghapus video temporer: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

current_video_second = 0.0

@app.get(api_test_video_metadata)
async def get_test_video_metadata():
    global video_source_path
    if not video_source_path or not os.path.exists(video_source_path):
        return JSONResponse({"status": "error", "message": "Video tidak ditemukan"}, status_code=400)
    
    cap = cv2.VideoCapture(video_source_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return {
        "status": "success",
        "duration": duration,
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height
    }

@app.get(api_test_video_stream)
async def stream_processed_video(start_second: float = 0.0):
    global video_source_path
    if not video_source_path or not os.path.exists(video_source_path):
        return JSONResponse({"status": "error", "message": "Video tidak ditemukan di disk"}, status_code=400)

    global video_detection_events, current_video_second
    if start_second == 0.0:
        video_detection_events.clear()

    def gen_frames():
        global current_video_second
        cap = cv2.VideoCapture(video_source_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Seek ke detik tertentu
        start_frame = int(start_second * fps)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            video_detection_events.append(f"[SYSTEM] Seek ke detik {start_second:.1f} (Frame {start_frame})...")
        else:
            video_detection_events.append("[SYSTEM] Membuka video via OpenCV Engine...")
            video_detection_events.append(f"[SYSTEM] Metadata Terbaca: Resolusi {width}x{height} | Total {total_frames} Frames | {fps:.1f} FPS")
            video_detection_events.append("[SYSTEM] Memulai deteksi wajah MTCNN & klasifikasi MobileFaceNet...")

        frame_idx = start_frame
        DEVICE = torch.device("cpu")
        
        face_buffers = {}  # Dictionary untuk temporal voting buffer masing-masing wajah (Key: track_id)
        next_track_id = 0
        prev_gray = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            # Kurangi frame processing rate (e.g. process setiap 2 frame agar tidak delay pada CPU lambat, tapi tampilkan semua)
            if frame_idx % 2 == 0:
                # Optimasi deteksi gerakan ringan
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (15, 15), 0)
                
                has_motion = True
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]
                    motion_ratio = np.sum(thresh) / (frame.shape[0] * frame.shape[1] * 255)
                    if motion_ratio < 0.003:  # Threshold 0.3% piksel berubah (tanda tidak ada gerakan)
                        has_motion = False
                prev_gray = gray
                
                if has_motion:
                    # BGR to RGB PIL
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    
                    # Deteksi seluruh wajah di dalam frame (Multi-Face)
                    detected_faces = image_processor.detect_face(img_pil, keep_all=True)
                    
                    current_face_centers = []
                    for face_tensor, box, prob in detected_faces:
                        x1, y1, x2, y2 = [int(b) for b in box]
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        current_face_centers.append((face_tensor, box, prob, center))
                    
                    # Asosiasikan wajah saat ini dengan tracking buffer sebelumnya (Threshold: 120px)
                    matched_faces = []
                    temp_buffers = {}
                    
                    for face_tensor, box, prob, center in current_face_centers:
                        best_id = None
                        min_dist = 120.0
                        
                        for track_id, info in face_buffers.items():
                            dist = np.sqrt((center[0] - info["center"][0])**2 + (center[1] - info["center"][1])**2)
                            if dist < min_dist:
                                min_dist = dist
                                best_id = track_id
                                
                        if best_id is not None:
                            track_id = best_id
                            p_buffer = face_buffers[track_id]["buffer"]
                            last_pred = face_buffers[track_id].get("last_pred", None)
                        else:
                            track_id = next_track_id
                            next_track_id += 1
                            p_buffer = deque(maxlen=5)
                            last_pred = None
                            
                        temp_buffers[track_id] = {
                            "center": center,
                            "buffer": p_buffer,
                            "last_seen": frame_idx,
                            "last_pred": last_pred
                        }
                        matched_faces.append((face_tensor, box, prob, track_id))
                        
                    # Pertahankan wajah lama yang hilang sementara (hang-time maks 5 frame)
                    for track_id, info in face_buffers.items():
                        if track_id not in temp_buffers and frame_idx - info["last_seen"] < 5:
                            temp_buffers[track_id] = info
                            
                    face_buffers = temp_buffers
                    
                    # Lakukan inferensi untuk wajah baru yang terdeteksi
                    for face_tensor, box, prob, track_id in matched_faces:
                        x1, y1, x2, y2 = [int(b) for b in box]
                        threshold = cl_client.threshold

                        # Siapkan input dan jalankan inferensi model
                        face_tensor_ready = image_processor.prepare_for_model(face_tensor)
                        with torch.no_grad():
                            cl_client.model.eval()
                            # Proses flip trick alignment dengan registry
                            emb_orig = cl_client.model(face_tensor_ready)
                            face_flipped = torch.flip(face_tensor_ready, dims=[3])
                            emb_mirror = cl_client.model(face_flipped)
                            
                            # Rata-rata dan normalisasi akhir ke unit vector (Flip-Only ITA)
                            query_emb_tensor = F.normalize((emb_orig + emb_mirror) / 2, p=2, dim=1).view(1, -1)
                            
                            p_buffer = face_buffers[track_id]["buffer"]
                            p_buffer.append(query_emb_tensor)
                            
                            # Temporal Average per wajah
                            mean_emb_tensor = torch.stack(list(p_buffer)).mean(0)
                            mean_emb_tensor = F.normalize(mean_emb_tensor, p=2, dim=1)

                        # Match
                        user_ids = list(cl_client.reference_embeddings.keys())
                        ref_list = []
                        for nrp in user_ids:
                            ref = cl_client.reference_embeddings[nrp]
                            if not isinstance(ref, torch.Tensor):
                                ref = torch.tensor(ref).to(DEVICE)
                            ref_list.append(ref.view(1, -1))
                        
                        if ref_list:
                            ref_matrix = torch.cat(ref_list, dim=0)
                            ref_matrix = F.normalize(ref_matrix, p=2, dim=1)
                            scores = torch.mm(mean_emb_tensor, ref_matrix.t())
                            max_sim, max_idx = torch.max(scores, dim=1)
                            sim = max_sim.item()
                            best_match = user_ids[max_idx.item()]
                            
                            is_confirmed = sim >= threshold
                            
                            # Cache prediksi terakhir
                            face_buffers[track_id]["last_pred"] = {
                                "best_match": best_match,
                                "sim": sim,
                                "is_confirmed": is_confirmed,
                                "box": box
                            }
                            
                            if is_confirmed:
                                nrp_only = best_match.split("_")[0] if "_" in best_match else best_match
                                video_detection_events.append(f"[Frame {frame_idx}] [CONFIRMED] {nrp_only} Teridentifikasi (Similarity: {sim:.2f})")
                            else:
                                if sim > 0.2:
                                    nrp_only = best_match.split("_")[0] if "_" in best_match else best_match
                                    video_detection_events.append(f"[Frame {frame_idx}] [REJECTED] Unknown Terdeteksi (Terbaik: {nrp_only} dengan Sim: {sim:.2f} < {threshold:.2f})")
                
                # Gambar seluruh bounding box aktif (baik yang baru dihitung maupun dari cache saat motion-skip!)
                for track_id, info in face_buffers.items():
                    if info.get("last_pred", None) is not None and frame_idx - info["last_seen"] < 5:
                        pred = info["last_pred"]
                        box = pred["box"]
                        is_confirmed = pred["is_confirmed"]
                        best_match = pred["best_match"]
                        sim = pred["sim"]
                        
                        x1, y1, x2, y2 = [int(b) for b in box]
                        color = (0, 255, 0) if is_confirmed else (0, 0, 255)
                        nrp_only = best_match.split("_")[0] if "_" in best_match else best_match
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        label = f"{nrp_only} ({sim:.2f})" if is_confirmed else f"Unknown: {nrp_only} ({sim:.2f})"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Update detik video saat ini
            current_video_second = frame_idx / fps
 
            # Compress to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.015)
            
        cap.release()
        video_detection_events.append("[SELESAI] Pemrosesan video selesai.")

    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get(api_test_video_events)
async def get_test_video_events():
    global video_detection_events, current_video_second
    return {
        "events": video_detection_events,
        "current_second": current_video_second
    }

# Event saat Startup
@app.on_event("startup")
def startup_event():
    cl_client.logger.info(f"Inisialisasi Client: {cl_client.client_id}")
    cl_client.start_background_tasks()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, access_log=False)