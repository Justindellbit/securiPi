import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import json
import os

WHITELIST_PATH = '/home/justin/compute/whitelist.json'
YUNET_PATH = '/home/justin/compute/face_detection_yunet_2023mar.onnx'

class FaceIdentifier:
    def __init__(self, model_path='mobilefacenet.tflite', threshold=0.6):
        self.threshold = threshold

        # YuNet
        self.yunet = cv2.FaceDetectorYN.create(
            YUNET_PATH, "", (320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5
        )
        print("[FaceID] YuNet chargé")

        # MobileFaceNet
        self.interpreter = Interpreter(model_path=model_path, num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = self.input_details[0]['shape'][1]
        print(f"[FaceID] MobileFaceNet chargé — input size : {self.input_size}")

        self.whitelist = self._load_whitelist()
        print(f"[FaceID] Whitelist : {list(self.whitelist.keys())}")

    def _load_whitelist(self):
        if os.path.exists(WHITELIST_PATH):
            with open(WHITELIST_PATH) as f:
                data = json.load(f)
            return {name: np.array(vec) for name, vec in data.items()}
        return {}

    def save_whitelist(self):
        data = {name: vec.tolist() for name, vec in self.whitelist.items()}
        with open(WHITELIST_PATH, 'w') as f:
            json.dump(data, f)

    def detect_face(self, frame):
        """Détecte le plus grand visage avec YuNet"""
        h, w = frame.shape[:2]
        if w < 20 or h < 20:
            return None

        # Upscale si le crop est trop petit
        scale = 1.0
        if w < 80 or h < 80:
            scale = max(80/w, 80/h)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            h, w = frame.shape[:2]

        self.yunet.setInputSize((w, h))
        _, faces = self.yunet.detect(frame)

        if faces is None or len(faces) == 0:
            return None

        # Prendre le visage avec le meilleur score
        best = max(faces, key=lambda f: f[14])
        x, y, fw, fh = int(best[0]), int(best[1]), int(best[2]), int(best[3])
        x, y = max(0, x), max(0, y)
        face_crop = frame[y:y+fh, x:x+fw]

        if face_crop.size == 0:
            return None

        # Remettre à l'échelle originale si upscalé
        if scale != 1.0:
            face_crop = cv2.resize(face_crop, (int(fw/scale), int(fh/scale)))

        return face_crop

    def encode_face(self, face_crop):
        """Transforme un crop de visage en vecteur 128D"""
        img = cv2.resize(face_crop, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - 127.5) / 128.0
        img = np.expand_dims(img, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        embedding = self.interpreter.get_tensor(self.output_details[0]['index'])
        return embedding.flatten()

    def enroll(self, name, frame):
        """Enregistre une personne dans la liste blanche"""
        face = self.detect_face(frame)
        if face is None:
            print(f"[FaceID] Aucun visage détecté pour {name}")
            return False
        embedding = self.encode_face(face)
        self.whitelist[name] = embedding
        self.save_whitelist()
        print(f"[FaceID] {name} enregistré — {len(embedding)} dimensions")
        return True

    def identify(self, person_crop):
        """Identifie une personne depuis le crop YOLO"""
        face = self.detect_face(person_crop)
        if face is None:
            return None, None

        embedding = self.encode_face(face)

        if not self.whitelist:
            return "UNKNOWN", None

        best_name = None
        best_dist = float('inf')
        for name, known_emb in self.whitelist.items():
            dist = np.linalg.norm(embedding - known_emb)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_dist < self.threshold:
            return best_name, best_dist
        return "UNKNOWN", best_dist
