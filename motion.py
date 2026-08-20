import cv2
import numpy as np

class MotionDetector:
    def __init__(self, min_area=500, sensitivity=50):
        self.min_area = min_area
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=sensitivity,
            detectShadows=True
        )

    def detect(self, frame):
        # Réduire la résolution pour accélérer MOG2
        small = cv2.resize(frame, (320, 240))
        mask = self.bg_sub.apply(small)

        # Supprimer les ombres (pixels gris = 127)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        # Nettoyer le bruit
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Trouver les contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        zones = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Remettre à l'échelle 640x480
            zones.append({
                'x': x * 2, 'y': y * 2,
                'w': w * 2, 'h': h * 2,
                'area': area * 4
            })

        return len(zones) > 0, zones
