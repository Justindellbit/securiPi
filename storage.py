import sqlite3
import cv2
import os
import time
from datetime import datetime

DB_PATH = '/home/justin/compute/securicam.db'
SNAPSHOTS_DIR = '/home/justin/compute/snapshots'

def init_db():
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS alarms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            camera TEXT NOT NULL,
            zone TEXT NOT NULL,
            level TEXT NOT NULL,
            class TEXT,
            confidence REAL,
            snapshot_path TEXT,
            acknowledged INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()
    print("[DB] Base initialisée")

def save_alarm(camera, zone, level, detections, frame):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ts_file = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Sauvegarder le snapshot annoté
    snapshot_path = None
    if frame is not None:
        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d['box']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{d['class']} {d['confidence']:.0%}"
            cv2.putText(annotated, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        snapshot_path = f"{SNAPSHOTS_DIR}/{ts_file}_{camera}_{zone}.jpg"
        cv2.imwrite(snapshot_path, annotated)

    # Enregistrer en DB
    top = detections[0] if detections else {}
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO alarms (timestamp, camera, zone, level, class, confidence, snapshot_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp, camera, zone, level,
        top.get('class'), top.get('confidence'),
        snapshot_path
    ))
    conn.commit()
    conn.close()
    print(f"[DB] Alarme sauvegardée : {timestamp} {level} {top.get('class')} → {snapshot_path}")
    return snapshot_path

def get_recent_alarms(limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM alarms ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return rows
