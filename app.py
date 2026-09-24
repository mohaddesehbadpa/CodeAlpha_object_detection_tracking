"""
app.py - Streamlit desktop UI for the Object Detection & Tracking project.

Run with:  streamlit run app.py

Flow:
  1. User uploads a video from their desktop.
  2. User picks a mode in the sidebar:
       - Track Everything      : detect + track every object class YOLO knows
       - Suspicious Activity   : flags loitering / running / crowding / abandoned bags
       - Find a Specific Person: upload a reference photo, app finds & zooms to that
                                  person whenever they appear
  3. Video is processed frame by frame; annotated frames stream live in the UI.
  4. The full annotated video is also written to disk and offered as a download.
"""

import tempfile
import time

import cv2
import numpy as np
import streamlit as st

from detector import ObjectDetector
from tracker import MultiObjectTracker
from person_reid import PersonReID
from suspicious_activity import SuspiciousActivityMonitor

st.set_page_config(page_title="Object Detection & Tracking", layout="wide")

# ---------- cached heavy objects so they load once per session ----------
@st.cache_resource
def load_detector():
    return ObjectDetector(model_path="yolov8n.pt", conf_threshold=0.30)

@st.cache_resource
def load_reid():
    return PersonReID()


def new_tracker(fast_mode=False):
    return MultiObjectTracker(fast_mode=fast_mode)


def draw_box(frame, bbox, color, text):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)


def make_zoom_inset(frame, bbox, inset_size=220):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return frame
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame
    zoom = cv2.resize(crop, (inset_size, inset_size))
    cv2.rectangle(zoom, (0, 0), (inset_size - 1, inset_size - 1), (0, 0, 255), 4)
    h, w = frame.shape[:2]
    frame[10:10 + inset_size, w - inset_size - 10: w - 10] = zoom
    cv2.putText(frame, "MATCH - ZOOM", (w - inset_size - 8, inset_size + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


# ---------------------------- Sidebar UI ----------------------------
st.sidebar.title("Settings")
mode = st.sidebar.radio(
    "What do you want to do?",
    ["Track Everything", "Suspicious Activity Monitor", "Find a Specific Person"],
)

video_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

reference_image_file = None
similarity_threshold = 0.75
if mode == "Find a Specific Person":
    reference_image_file = st.sidebar.file_uploader(
        "Upload a photo of the person to find", type=["jpg", "jpeg", "png"]
    )
    similarity_threshold = st.sidebar.slider(
        "Match sensitivity (higher = stricter)", 0.5, 0.95, 0.75, 0.01
    )

conf_threshold = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.30, 0.05)

with st.sidebar.expander("Performance (CPU / no GPU tips)", expanded=True):
    resize_width = st.select_slider(
        "Downscale width before detection (px)",
        options=[320, 480, 640, 960, 1280],
        value=640,
        help="Smaller = much faster detection, slightly less accurate on tiny objects.",
    )
    frame_skip = st.slider(
        "Process every Nth frame", 1, 5, 2,
        help="2 = detect on every other frame and reuse the last boxes in between. "
             "Big speedup on CPU with only a small drop in tracking smoothness.",
    )
    fast_tracking = st.checkbox(
        "Fast tracking (skip appearance model)", value=True,
        help="Tracks by motion only instead of also running a neural net per box "
             "per frame. Much faster on CPU; IDs may swap more when objects cross.",
    )
    show_live_preview = st.checkbox(
        "Show live preview while processing", value=True,
        help="Turning this off skips re-rendering every frame in the browser and "
             "just gives you the finished video at the end — noticeably faster.",
    )
    preview_interval = st.slider(
        "Live preview update interval", 1, 15, 5,
        help="Update the browser preview less often to avoid slowing processing.",
    )

run_button = st.sidebar.button("Start processing", type="primary")

st.title("Real-Time Object Detection & Tracking")

frame_placeholder = st.empty()
alert_placeholder = st.empty()
progress_placeholder = st.empty()

# ---------------------------- Main processing ----------------------------
if run_button:
    if video_file is None:
        st.warning("Upload a video first.")
        st.stop()
    if mode == "Find a Specific Person" and reference_image_file is None:
        st.warning("Upload a reference photo of the person to search for.")
        st.stop()

    detector = load_detector()
    detector.conf_threshold = conf_threshold
    tracker = new_tracker(fast_mode=fast_tracking)

    reid = None
    reference_embedding = None
    if mode == "Find a Specific Person":
        reid = load_reid()
        file_bytes = np.frombuffer(reference_image_file.read(), np.uint8)
        ref_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # detect a person in the reference photo so we embed just that region
        ref_detections = detector.detect(ref_img)
        person_boxes = [d for d in ref_detections if d["label"] == "person"]
        if person_boxes:
            x1, y1, x2, y2 = [int(v) for v in person_boxes[0]["bbox"]]
            ref_crop = ref_img[max(y1, 0):y2, max(x1, 0):x2]
        else:
            ref_crop = ref_img  # fall back to whole image
        reference_embedding = reid.embed(ref_crop)
        st.sidebar.image(cv2.cvtColor(ref_crop, cv2.COLOR_BGR2RGB), caption="Searching for this person", width=150)

    monitor = None
    if mode == "Suspicious Activity Monitor":
        monitor = SuspiciousActivityMonitor()

    # save upload to a temp file so OpenCV can open it
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if monitor:
        monitor.fps = fps

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = None

    frame_idx = 0
    all_alerts_log = []
    last_tracks = []
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # downscale before detection for speed; boxes are scaled back up
        h0, w0 = frame.shape[:2]
        if w0 > resize_width:
            scale = resize_width / w0
            small = cv2.resize(frame, (resize_width, int(h0 * scale)))
        else:
            scale = 1.0
            small = frame

        if frame_idx % frame_skip == 0:
            detections = detector.detect(small)
            if scale != 1.0:
                for d in detections:
                    d["bbox"] = [v / scale for v in d["bbox"]]
            tracks = tracker.update(detections, frame)
            last_tracks = tracks
        else:
            # reuse the previous frame's tracks instead of re-running
            # detection - keeps things smooth without the full cost
            tracks = last_tracks

        if mode == "Track Everything":
            for t in tracks:
                draw_box(frame, t["bbox"], (0, 200, 0), f'{t["label"]} #{t["track_id"]}')

        elif mode == "Suspicious Activity Monitor":
            for t in tracks:
                draw_box(frame, t["bbox"], (0, 200, 0), f'{t["label"]} #{t["track_id"]}')
            alerts = monitor.update(tracks, frame_idx)
            for a in alerts:
                if "bbox" in a:
                    draw_box(frame, a["bbox"], (0, 0, 255), a["type"].upper())
                all_alerts_log.append({"frame": frame_idx, **a})
            if alerts:
                alert_placeholder.warning(
                    "  |  ".join(f'{a["type"]}: {a.get("detail","")}' for a in alerts)
                )

        elif mode == "Find a Specific Person":
            best_match, best_score = None, -1.0
            for t in tracks:
                if t["label"] != "person":
                    draw_box(frame, t["bbox"], (150, 150, 150), t["label"])
                    continue
                x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
                crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                emb = reid.embed(crop)
                score = reid.cosine_similarity(emb, reference_embedding)
                if score > similarity_threshold:
                    draw_box(frame, t["bbox"], (0, 0, 255), f"MATCH ({score:.2f})")
                    if score > best_score:
                        best_score, best_match = score, t["bbox"]
                else:
                    draw_box(frame, t["bbox"], (0, 200, 0), f'person #{t["track_id"]}')
            if best_match is not None:
                frame = make_zoom_inset(frame, best_match)
                alert_placeholder.success(f"Person found — similarity {best_score:.2f}")
            else:
                alert_placeholder.info("Searching...")

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)

        # Streamlit reruns the frontend on each update, so frequent image and
        # progress updates can become the bottleneck on longer videos.
        should_update_ui = frame_idx == 1 or frame_idx % preview_interval == 0
        if show_live_preview and should_update_ui:
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        if total_frames and should_update_ui:
            elapsed = time.time() - start_time
            live_fps = frame_idx / elapsed if elapsed > 0 else 0
            progress_placeholder.progress(
                min(frame_idx / total_frames, 1.0),
                text=f"Frame {frame_idx}/{total_frames} — {live_fps:.1f} fps",
            )

    cap.release()
    if writer:
        writer.release()

    st.success("Done processing.")
    with open(out_path, "rb") as f:
        st.download_button("Download annotated video", f, file_name="annotated_output.mp4")

    if mode == "Suspicious Activity Monitor" and all_alerts_log:
        st.subheader("Alert log")
        st.dataframe(all_alerts_log)