"""
tracker.py
Wraps deep-sort-realtime so every detected object (not just people) gets a
persistent track ID across frames, using appearance + motion matching.
"""

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class MultiObjectTracker:
    def __init__(self, max_age=30, n_init=2, fast_mode=False):
        """
        fast_mode=True drops DeepSORT's appearance embedder (the MobileNetv2
        forward pass run on every box, every frame) and tracks on motion
        only. Much faster on CPU; IDs may swap more easily when objects
        cross paths or briefly overlap.

        Note: when embedder=None, deep-sort-realtime requires *some*
        per-detection feature vector to be supplied on every update call
        (it won't silently skip appearance matching) - so in fast_mode we
        hand it identical dummy vectors, which makes appearance distance a
        no-op and effectively falls back to motion-only (Kalman/IoU) matching.
        """
        self.fast_mode = fast_mode
        embedder = None if fast_mode else "mobilenet"
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=1.0,
            embedder=embedder,
        )

    def update(self, detections, frame):
        """
        detections: output of ObjectDetector.detect()
        Returns list of dicts: {"track_id": str, "bbox": [x1,y1,x2,y2], "label": str}
        """
        ds_input = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            w, h = x2 - x1, y2 - y1
            ds_input.append(([x1, y1, w, h], d["confidence"], d["label"]))

        if self.fast_mode:
            # identical, non-zero dummy embeddings -> cosine distance between
            # any two of them is well-defined (no divide-by-zero) and always
            # the same value, so appearance contributes nothing and matching
            # falls back to motion/IoU only
            dummy_embeds = [np.ones(1, dtype=np.float32) for _ in ds_input]
            tracks = self.tracker.update_tracks(ds_input, embeds=dummy_embeds)
        else:
            tracks = self.tracker.update_tracks(ds_input, frame=frame)

        results = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = t.to_ltrb()
            label = t.get_det_class() if hasattr(t, "get_det_class") and t.get_det_class() else "object"
            results.append({
                "track_id": str(t.track_id),
                "bbox": [x1, y1, x2, y2],
                "label": label,
            })
        return results