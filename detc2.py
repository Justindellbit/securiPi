import numpy as np
import cv2
import time
from ai_edge_litert.interpreter import Interpreter

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class YOLODetector:
    def __init__(self, model_path='best_int8.tflite', conf_threshold=0.25):
        self.conf_threshold = conf_threshold
        self.interpreter = Interpreter(
            model_path=model_path,
            num_threads=4
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.input_size = self.input_shape[1]
        print(f"[YOLO] Modèle chargé : {model_path}")
        print(f"[YOLO] Input shape : {self.input_shape}")
        print(f"[YOLO] Type : {self.input_details[0]['dtype']}")

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        if self.input_details[0]['dtype'] == np.uint8:
            return img.astype(np.uint8)
        return img.astype(np.float32) / 255.0

    def postprocess(self, output, orig_w, orig_h):
        output_details = self.output_details[0]
        if output_details['dtype'] == np.int8:
            scale, zero_point = output_details['quantization']
            output = (output.astype(np.float32) - zero_point) * scale

        predictions = output[0].T  # (2100, 84)
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        mask = confidences >= self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        results = []
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            cx, cy, w, h = box
            # Coordonnées normalisées → pixels
            x1 = int((cx - w/2) * orig_w)
            y1 = int((cy - h/2) * orig_h)
            x2 = int((cx + w/2) * orig_w)
            y2 = int((cy + h/2) * orig_h)
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            if x2 > x1 and y2 > y1:
                results.append({
                    'class': CLASSES[cls_id] if cls_id < len(CLASSES) else f'cls_{cls_id}',
                    'confidence': float(conf),
                    'box': [x1, y1, x2, y2]
                })

        if results:
            boxes_nms = [[r['box'][0], r['box'][1], r['box'][2]-r['box'][0], r['box'][3]-r['box'][1]] for r in results]
            scores_nms = [r['confidence'] for r in results]
            indices = cv2.dnn.NMSBoxes(boxes_nms, scores_nms, self.conf_threshold, 0.4)
            if len(indices) > 0:
                results = [results[i] for i in indices.flatten()]

        return results

    def detect(self, frame):
        orig_h, orig_w = frame.shape[:2]
        t0 = time.time()
        inp = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        results = self.postprocess(output, orig_w, orig_h)
        ms = (time.time() - t0) * 1000
        return results, ms
