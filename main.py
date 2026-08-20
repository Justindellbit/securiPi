import sys, time, signal
sys.path.insert(0, '/home/justin/compute')

from camera import CameraStream
from motion import MotionDetector
from detector import YOLODetector
from alarm_engine import AlarmEngine
from storage import init_db, save_alarm
from notifier import TelegramNotifier
import cv2

# ===== CONFIGURATION =====
CAMERA_URL = "http://192.168.137.123:8080/video?640x480"
TELEGRAM_TOKEN = "8869372969:AAFtd2RMojnltqi1Wx1QFraLiPZ61Xt-cec"
TELEGRAM_CHAT_ID = "6236147298"
ZONES_CONFIG = '/home/justin/compute/zones.json'
MODEL_PATH = '/home/justin/compute/best_int8.tflite'
PROCESS_SIZE = 320
# ==========================

running = True

def signal_handler(sig, frame):
    global running
    print("\n[Main] Arrêt demandé...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

def main():
    print("=== SecuriCam-Pi — Démarrage ===\n")

    init_db()

    cam = CameraStream(src=CAMERA_URL)
    motion = MotionDetector(min_area=500)
    yolo = YOLODetector(MODEL_PATH)
    engine = AlarmEngine(ZONES_CONFIG)
    notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)

    cam.start()
    print("[Main] Attente connexion caméra...")
    time.sleep(5)
    print("[Main] Système actif — surveillance en cours\n")

    frame_count = 0
    alarm_count = 0
    start_time = time.time()

    while running:
        ts, frame = cam.get_latest()
        if frame is None:
            time.sleep(0.05)
            continue

        frame_count += 1
        small = cv2.resize(frame, (PROCESS_SIZE, PROCESS_SIZE))

        has_motion, zones = motion.detect(small)

        if has_motion:
            detections, ms = yolo.detect(small)
            alarms = engine.process(0, zones)

            for alarm in alarms:
                alarm_count += 1
                print(f"[ALARME #{alarm_count}] {alarm['level']} — "
                      f"{alarm['zone']} — {time.strftime('%H:%M:%S')}")

                snapshot_path = save_alarm(
                    alarm['cam'], alarm['zone'], alarm['level'],
                    detections, frame
                )

                alarm['class'] = detections[0]['class'] if detections else 'mouvement'
                alarm['confidence'] = f"{detections[0]['confidence']:.0%}" if detections else ''
                notifier.notify(alarm, snapshot_path)

        time.sleep(0.1)

        # Stats toutes les 60s
        if frame_count % 600 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"[Stats] {frame_count} frames | {fps:.1f} FPS | {alarm_count} alarmes")

    cam.stop()
    print(f"\n[Main] Arrêté — {frame_count} frames traitées, {alarm_count} alarmes")

if __name__ == '__main__':
    main()
