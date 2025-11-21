import cv2
import math
from ultralytics import YOLO
from models import db, CountEvent

class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0
        self.objects = {}  # id -> (cx, cy)
        self.max_distance = max_distance

    def update(self, detections):
        if len(self.objects) == 0:
            for cx, cy in detections:
                self.objects[self.next_id] = (cx, cy)
                self.next_id += 1
            return self.objects

        new_objects = {}
        used_ids = set()
        for (cx, cy) in detections:
            best_id = None
            best_dist = float("inf")
            for obj_id, (ox, oy) in self.objects.items():
                dist = math.hypot(cx - ox, cy - oy)
                if dist < best_dist and dist < self.max_distance and obj_id not in used_ids:
                    best_dist = dist
                    best_id = obj_id

            if best_id is None:
                new_objects[self.next_id] = (cx, cy)
                used_ids.add(self.next_id)
                self.next_id += 1
            else:
                new_objects[best_id] = (cx, cy)
                used_ids.add(best_id)

        self.objects = new_objects
        return self.objects

class PeopleCounter:
    def __init__(self, camera_index=0):
        self.model = YOLO("yolov8n.pt")
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.tracker = CentroidTracker(max_distance=60)
        self.in_count = 0
        self.out_count = 0
        self.lobby_count = 0
        self.last_positions = {}  # id -> last cx

    def _log_event(self, direction):
        """Save IN/OUT event to database."""
        if direction == "IN":
            self.lobby_count += 1
        elif direction == "OUT" and self.lobby_count > 0:
            self.lobby_count -= 1

        event = CountEvent(direction=direction, lobby_count=self.lobby_count)
        db.session.add(event)
        db.session.commit()

    def generate_frames(self):
        """Generator that yields JPEG frames with boxes and counters drawn."""
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            h, w = frame.shape[:2]
            line_x = w // 2  # vertical line in middle

            # YOLO person detection
            results = self.model(frame, classes=[0], verbose=False)

            detections = []
            boxes_list = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    if conf < 0.4:
                        continue
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections.append((cx, cy))
                    boxes_list.append((x1, y1, x2, y2, cx, cy))

            objects = self.tracker.update([(cx, cy) for (cx, cy) in detections])

            # Draw vertical line
            cv2.line(frame, (line_x, 0), (line_x, h), (255, 0, 0), 2)

            # Associate boxes with IDs
            for (x1, y1, x2, y2, cx, cy) in boxes_list:
                obj_id = None
                min_dist = float("inf")
                for tid, (tx, ty) in objects.items():
                    d = math.hypot(cx - tx, cy - ty)
                    if d < min_dist and d < 30:
                        min_dist = d
                        obj_id = tid

                if obj_id is not None:
                    last_cx = self.last_positions.get(obj_id, cx)
                    # left to right
                    if last_cx < line_x and cx >= line_x:
                        self.in_count += 1
                        self._log_event("IN")
                    # right to left
                    elif last_cx > line_x and cx <= line_x:
                        self.out_count += 1
                        self._log_event("OUT")

                    self.last_positions[obj_id] = cx

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    label = f"ID {obj_id}"
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw counts
            cv2.putText(frame, f"IN: {self.in_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"OUT: {self.out_count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"In Lobby: {self.lobby_count}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield as HTTP multipart
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
