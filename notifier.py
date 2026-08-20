import threading
import time
import requests

class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.queue = []
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _send_photo(self, snapshot_path, caption, retries=3):
        for attempt in range(retries):
            try:
                with open(snapshot_path, 'rb') as f:
                    resp = requests.post(
                        f"{self.base_url}/sendPhoto",
                        data={'chat_id': self.chat_id, 'caption': caption},
                        files={'photo': f},
                        timeout=10
                    )
                if resp.status_code == 200:
                    print(f"[Telegram] Photo envoyée OK")
                    return True
                else:
                    print(f"[Telegram] Erreur {resp.status_code}")
            except Exception as e:
                print(f"[Telegram] Tentative {attempt+1} échouée : {e}")
                time.sleep(2 ** attempt)
        return False

    def _send_text(self, message, retries=3):
        for attempt in range(retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/sendMessage",
                    data={'chat_id': self.chat_id, 'text': message},
                    timeout=10
                )
                if resp.status_code == 200:
                    return True
            except Exception as e:
                print(f"[Telegram] Tentative {attempt+1} échouée : {e}")
                time.sleep(2 ** attempt)
        return False

    def _worker(self):
        while True:
            task = None
            with self.lock:
                if self.queue:
                    task = self.queue.pop(0)
            if task:
                if task['type'] == 'photo':
                    self._send_photo(task['path'], task['caption'])
                elif task['type'] == 'text':
                    self._send_text(task['message'])
            else:
                time.sleep(0.2)

    def notify(self, alarm, snapshot_path=None):
        caption = (
            f"🚨 ALARME {alarm['level']}\n"
            f"📷 {alarm['cam']} — zone {alarm['zone']}\n"
            f"🕐 {time.strftime('%H:%M:%S')}\n"
            f"🎯 {alarm.get('class', 'mouvement')} {alarm.get('confidence', '')}"
        )
        with self.lock:
            if snapshot_path:
                self.queue.append({
                    'type': 'photo',
                    'path': snapshot_path,
                    'caption': caption
                })
            else:
                self.queue.append({
                    'type': 'text',
                    'message': caption
                })
