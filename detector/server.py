"""
Web server for remote camera feed and controls.
"""

import threading
import time
import os
from typing import Optional

import cv2
import numpy as np

try:
    from flask import Flask, Response, render_template, jsonify, request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

from .config import DetectorConfig

app = None
if HAS_FLASK:
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))


class SharedState:
    """Thread-safe state shared between detector loop and web server."""
    def __init__(self) -> None:
        self.latest_frame: Optional[np.ndarray] = None
        self.fps: float = 0.0
        self.face_count: int = 0
        self.cfg: Optional[DetectorConfig] = None
        self.recording: bool = False
        self.lock = threading.Lock()

    def update_frame(self, frame: np.ndarray) -> None:
        with self.lock:
            self.latest_frame = frame.copy()

    def update_stats(self, fps: float, face_count: int, recording: bool) -> None:
        with self.lock:
            self.fps = fps
            self.face_count = face_count
            self.recording = recording

    def set_config(self, cfg: DetectorConfig) -> None:
        with self.lock:
            self.cfg = cfg


shared_state = SharedState()


if HAS_FLASK:
    @app.route('/')
    def index():
        return render_template('index.html')

    def generate_mjpeg_stream():
        while True:
            with shared_state.lock:
                frame = shared_state.latest_frame
            
            if frame is None:
                time.sleep(0.03)
                continue

            # Encode frame to JPEG
            ret, encoded_img = cv2.imencode('.jpg', frame)
            if not ret:
                time.sleep(0.03)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_img.tobytes() + b'\r\n')
            time.sleep(0.04)  # ~25 FPS stream limit to save bandwidth

    @app.route('/stream')
    def stream():
        return Response(generate_mjpeg_stream(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/api/stats')
    def get_stats():
        with shared_state.lock:
            return jsonify({
                'fps': round(shared_state.fps, 1),
                'face_count': shared_state.face_count,
                'recording': shared_state.recording,
                'backend': shared_state.cfg.backend if shared_state.cfg else 'unknown'
            })

    @app.route('/api/config', methods=['GET', 'POST'])
    def handle_config():
        with shared_state.lock:
            cfg = shared_state.cfg
        
        if not cfg:
            return jsonify({'error': 'Config not initialized'}), 500

        if request.method == 'POST':
            data = request.json or {}
            if 'detect_faces' in data:
                cfg.detect_faces = bool(data['detect_faces'])
            if 'detect_eyes' in data:
                cfg.detect_eyes = bool(data['detect_eyes'])
            if 'detect_smiles' in data:
                cfg.detect_smiles = bool(data['detect_smiles'])
            if 'backend' in data:
                cfg.backend = str(data['backend'])

        return jsonify({
            'detect_faces': cfg.detect_faces,
            'detect_eyes': cfg.detect_eyes,
            'detect_smiles': cfg.detect_smiles,
            'backend': cfg.backend
        })


def start_web_server(cfg: DetectorConfig, host: str = "0.0.0.0", port: int = 5000) -> None:
    """Starts the Flask web server in a daemon thread."""
    if not HAS_FLASK:
        print("Flask is not installed. Web dashboard is unavailable.")
        return

    shared_state.set_config(cfg)
    
    server_thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True
    )
    server_thread.start()
    print(f"Web dashboard running on http://{host}:{port}")
