import sys, time, signal, cv2
sys.path.insert(0, '/home/justin/compute')

from camera import CameraStream
from motion import MotionDetector
from detector import YOLODetector
from alarm_engine import AlarmEngine
from face_id import FaceIdentifier
from storage import init_db, save_alarm
from notifier import TelegramNotifier
from collections import deque

# ===== CONFIGURATION =====
CAMERA_URL      = "http://192.168.137.19:8080/video?640x480"
TELEGRAM_TOKEN  = "8869372969:AAFtd2RMojnltqi1Wx1QFraLiPZ61Xt-cec"
TELEGRAM_CHAT_ID= "6236147298"
ZONES_CONFIG    = '/home/justin/compute/zones.json'
YOLO_MODEL      = '/home/justin/compute/best_int8.tflite'
FACE_MODEL      = '/home/justin/compute/mobilefacenet.tflite'
FACE_THRESHOLD  = 0.55
VOTE_FRAMES     = 5   # nb frames pour vote majoritaire
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

    cam      = CameraStream(src=CAMERA_URL)
    motion   = MotionDetector(min_area=500)
    yolo     = YOLODetector(YOLO_MODEL, conf_threshold=0.5)
    engine   = AlarmEngine(ZONES_CONFIG)
    face_id  = FaceIdentifier(FACE_MODEL, threshold=FACE_THRESHOLD)
    notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)

    cam.start()
    print("[Main] Attente connexion caméra...")
    time.sleep(5)
    print("[Main] Système actif — surveillance en cours\n")

    # Voting buffer : {person_id: deque de résultats}
    vote_buffer = deque(maxlen=VOTE_FRAMES)

    frame_count = 0
    alarm_count = 0
    start_time  = time.time()

    while running:
        ts, frame = cam.get_latest()
        if frame is None:
            time.sleep(0.05)
            continue

        frame_count += 1

        # === COUCHE 1 : MOG2 — gardien rapide ===
        small = cv2.resize(frame, (320, 240))
        has_motion, zones = motion.detect(small)

        if not has_motion:
            time.sleep(0.05)
            continue

        # === COUCHE 2 : YOLO — détection personnes ===
        detections, yolo_ms = yolo.detect(frame)
        persons = [d for d in detections if d['class'] == 'person']

        if not persons:
            time.sleep(0.05)
            continue

        # === COUCHE 3 : FaceID — identification ===
        for d in persons:
            x1, y1, x2, y2 = d['box']
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            name, dist = face_id.identify(person_crop)

            # Voting : accumuler les résultats
            vote_buffer.append(name)

            if len(vote_buffer) < VOTE_FRAMES:
                continue

            # Décision majoritaire
            known = [n for n in vote_buffer if n not in (None, "UNKNOWN")]
            unknown = [n for n in vote_buffer if n == "UNKNOWN"]

            if len(known) >= 3:
                winner = max(set(known), key=known.count)
                print(f"[FaceID] ✅ AUTORISÉ — {winner} ({len(known)}/{VOTE_FRAMES})")
                vote_buffer.clear()
                continue

            if len(unknown) >= 3:
                print(f"[FaceID] 🚨 INTRUS détecté ! ({len(unknown)}/{VOTE_FRAMES})")
                vote_buffer.clear()

                # === COUCHE 4 : Alarme CRITICAL ===
                alarm = {
                    'cam': 'cam0',
                    'zone': 'entree',
                    'level': 'CRITICAL',
                    'class': 'intrus',
                    'confidence': f"{d['confidence']:.0%}"
                }
                alarm_count += 1
                snapshot_path = save_alarm(
                    alarm['cam'], alarm['zone'], alarm['level'],
                    detections, frame
                )
                notifier.notify(alarm, snapshot_path)

            # Moteur de règles pour alarmes WARNING (mouvement sans identification)
            rule_alarms = engine.process(0, zones)
            for ralarm in rule_alarms:
                if ralarm['level'] == 'WARNING' and name is None:
                    ralarm['class'] = 'mouvement'
                    ralarm['confidence'] = ''
                    snapshot_path = save_alarm(
                        ralarm['cam'], ralarm['zone'], ralarm['level'],
                        detections, frame
                    )
                    notifier.notify(ralarm, snapshot_path)

        # Stats toutes les 60s
        if frame_count % 600 == 0:
            elapsed = time.time() - start_time
            print(f"[Stats] {frame_count} frames | "
                  f"{frame_count/elapsed:.1f} FPS effectif | "
                  f"{alarm_count} alarmes")

        time.sleep(0.1)

    cam.stop()
    print(f"\n[Main] Arrêté — {frame_count} frames | {alarm_count} alarmes")

if __name__ == '__main__':
    main()
