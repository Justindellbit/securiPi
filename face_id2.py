import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import json
import os

WHITELIST_PATH = '/home/justin/compute/whitelist.json'

class FaceIdentifier:
    def __init__(self, model_path='mobilefacenet.tflite', threshold=0.6):
        self.threshold = threshold
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.interpreter = Interpreter(model_path=model_path, num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = self.input_details[0]['shape'][1]
        print(f"[FaceID] Modèle chargé : {model_path}")
        print(f"[FaceID] Input size : {self.input_size}")

        self.whitelist = self._load_whitelist()

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
        """Détecte le plus grand visage dans une frame (ou crop de personne)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            return None
        # Prendre le plus grand visage détecté
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return frame[y:y+h, x:x+w]

    def encode_face(self, face_crop):
        """Transforme un crop de visage en vecteur d'embedding"""
        img = cv2.resize(face_crop, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - 127.5) / 128.0
        img = np.expand_dims(img, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        embedding = self.interpreter.get_tensor(self.output_details[0]['index'])
        return embedding.flatten()

    def enroll(self, name, frame):
        """Enregistre une nouvelle personne dans la liste blanche"""
        face = self.detect_face(frame)
        if face is None:
            print(f"[FaceID] Aucun visage détecté pour {name}")
            return False
        embedding = self.encode_face(face)
        self.whitelist[name] = embedding
        self.save_whitelist()
        print(f"[FaceID] {name} enregistré dans la liste blanche")
        return True

    def identify(self, person_crop):
        """Identifie une personne à partir du crop YOLO. Retourne (nom, distance) ou (None, None)"""
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
