import cv2
import threading
from collections import deque
import time
import psutil

class CameraStream:
    def __init__(self, src=0, width=640, height=480, fps=15, buffer_size=30):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer = deque(maxlen=buffer_size)
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.failures = 0
        self.MAX_FAILURES = 10

    def _open(self):
        cap = cv2.VideoCapture(self.src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def _capture_loop(self):
        cap = self._open()
        print(f"[Camera {self.src}] Démarré")

        while self.running:
            # Sous-échantillonnage adaptatif selon charge CPU
            cpu = psutil.cpu_percent(interval=None)
            skip = 3 if cpu > 80 else 2 if cpu > 60 else 1
            
            for _ in range(skip - 1):
                cap.grab()  # lit sans décoder (rapide)

            ret, frame = cap.read()

            if not ret:
                self.failures += 1
                print(f"[Camera {self.src}] Échec lecture ({self.failures}/{self.MAX_FAILURES})")
                if self.failures >= self.MAX_FAILURES:
                    print(f"[Camera {self.src}] Reconnexion...")
                    cap.release()
                    time.sleep(2)
                    cap = self._open()
                    self.failures = 0
                continue

            self.failures = 0
            with self.lock:
                self.buffer.append((time.time(), frame))

        cap.release()
        print(f"[Camera {self.src}] Arrêté")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)

    def get_latest(self):
        with self.lock:
            if self.buffer:
                return self.buffer[-1]
        return None, None
