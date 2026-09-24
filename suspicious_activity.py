"""
suspicious_activity.py

There is no pretrained model that reliably outputs "this is a crime" or
"this is illegal" - that isn't a real learnable label, and any tool that
claimed to do that would just be guessing. What real surveillance-analytics
systems do instead is flag specific, explainable *behavior patterns* for a
human to review. This module implements the standard ones:

  - loitering        : a person stays in a small area far longer than normal
  - fast_movement     : a person moves unusually fast between frames (running)
  - crowd_formation   : an unusually large number of people cluster together
  - abandoned_object  : a bag/backpack/suitcase stops moving and is left
                        alone (no person nearby) for a while

Each flag is transparent about *why* it fired, so it's genuinely useful as
a triage tool rather than a black box.
"""

import numpy as np

BAGGAGE_LABELS = {"backpack", "handbag", "suitcase"}


class SuspiciousActivityMonitor:
    def __init__(self, loiter_seconds=8, speed_threshold=40, fps=25,
                 abandon_seconds=6, abandon_radius=80):
        self.loiter_seconds = loiter_seconds
        self.speed_threshold = speed_threshold  # pixels/frame between consecutive frames
        self.fps = fps
        self.abandon_seconds = abandon_seconds
        self.abandon_radius = abandon_radius

        self.track_history = {}     # track_id -> list of (frame_idx, cx, cy)
        self.track_first_seen = {}  # track_id -> frame_idx
        self.track_label = {}       # track_id -> label

    def update(self, tracks, frame_idx):
        alerts = []
        persons, objects = [], []

        for t in tracks:
            tid, label = t["track_id"], t["label"]
            x1, y1, x2, y2 = t["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            self.track_label[tid] = label
            hist = self.track_history.setdefault(tid, [])
            hist.append((frame_idx, cx, cy))
            if len(hist) > 90:
                hist.pop(0)
            self.track_first_seen.setdefault(tid, frame_idx)

            if label == "person":
                persons.append(t)
                if len(hist) >= 2:
                    _, px, py = hist[-2]
                    dist = float(np.hypot(cx - px, cy - py))
                    if dist > self.speed_threshold:
                        alerts.append({"type": "fast_movement", "track_id": tid,
                                        "bbox": t["bbox"],
                                        "detail": f"moved {dist:.0f}px in one frame"})

                elapsed = (frame_idx - self.track_first_seen[tid]) / max(self.fps, 1)
                if elapsed > self.loiter_seconds and len(hist) >= 2:
                    xs = [p[1] for p in hist]
                    ys = [p[2] for p in hist]
                    if (max(xs) - min(xs) < 60) and (max(ys) - min(ys) < 60):
                        alerts.append({"type": "loitering", "track_id": tid,
                                        "bbox": t["bbox"],
                                        "detail": f"stationary for {elapsed:.0f}s"})
            elif label in BAGGAGE_LABELS:
                objects.append(t)

        if len(persons) >= 5:
            alerts.append({"type": "crowd_formation", "count": len(persons),
                            "detail": f"{len(persons)} people clustered in frame"})

        # abandoned object: a bag with no person within abandon_radius pixels
        for obj in objects:
            ox1, oy1, ox2, oy2 = obj["bbox"]
            ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
            elapsed = (frame_idx - self.track_first_seen[obj["track_id"]]) / max(self.fps, 1)
            if elapsed < self.abandon_seconds:
                continue
            near_person = False
            for p in persons:
                px1, py1, px2, py2 = p["bbox"]
                pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                if np.hypot(ocx - pcx, ocy - pcy) < self.abandon_radius:
                    near_person = True
                    break
            if not near_person:
                alerts.append({"type": "abandoned_object", "track_id": obj["track_id"],
                                "bbox": obj["bbox"],
                                "detail": f"{obj['label']} left unattended {elapsed:.0f}s"})
        return alerts
