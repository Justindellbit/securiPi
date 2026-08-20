import json
import time

class AlarmEngine:
    def __init__(self, rules_path='zones.json'):
        with open(rules_path) as f:
            self.config = json.load(f)
        # {cam_id: {zone: [timestamps des détections]}}
        self.detections = {}
        # {cam_id: {zone: last_alarm_time}}
        self.cooldowns = {}

    def _point_in_polygon(self, x, y, polygon):
        inside = False
        n = len(polygon)
        px, py = x, y
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and \
               (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def process(self, cam_id, zones):
        cam_key = f"cam{cam_id}"
        if cam_key not in self.config:
            return []

        now = time.time()
        alarms = []

        for zone_name, zone_cfg in self.config[cam_key].items():
            polygon = zone_cfg['polygon']
            rules = zone_cfg['rules']

            # Vérifier si un mouvement tombe dans cette zone
            zone_triggered = False
            for z in zones:
                cx = z['x'] + z['w'] // 2
                cy = z['y'] + z['h'] // 2
                if self._point_in_polygon(cx, cy, polygon):
                    if z['area'] >= rules['mouvement']['min_area']:
                        zone_triggered = True
                        break

            if not zone_triggered:
                continue

            # Anti-rebond : N détections en T secondes
            key = f"{cam_key}_{zone_name}"
            if key not in self.detections:
                self.detections[key] = []
            self.detections[key].append(now)

            t = rules['mouvement']['debounce_t']
            n = rules['mouvement']['debounce_n']
            # Garder seulement les détections récentes
            self.detections[key] = [
                ts for ts in self.detections[key] if now - ts <= t
            ]

            if len(self.detections[key]) < n:
                continue

            # Cooldown : éviter le spam
            if key not in self.cooldowns:
                self.cooldowns[key] = 0
            if now - self.cooldowns[key] < rules['mouvement']['cooldown']:
                continue

            self.cooldowns[key] = now
            self.detections[key] = []

            alarms.append({
                'cam': cam_key,
                'zone': zone_name,
                'level': rules['mouvement']['level'],
                'time': now
            })

        return alarms
