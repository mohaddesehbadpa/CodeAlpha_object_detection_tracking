# Real-Time Object Detection & Tracking (with UI)

Built for the internship task, extended with:
- A Streamlit UI where you pick a video from your desktop instead of hardcoding a path.
- **Track Everything** mode — detects and tracks all 80 COCO classes (person,
  ball, backpack, bottle, dog, chair, cell phone, etc.), so yes, a ball on the
  ground gets tracked as long as it's a decent number of pixels on screen.
- **Suspicious Activity Monitor** — flags loitering, sudden fast movement,
  crowd formation, and abandoned bags/suitcases, with a plain-language reason
  for each flag.
- **Find a Specific Person** — upload a reference photo, and any tracked
  person whose appearance matches gets outlined in red with a zoomed-in
  inset panel and a similarity score.

## Why there's no "illegal things / crime" detector

Being straightforward about this because it matters for how you present the
project: there is no pretrained model — YOLO or otherwise — that outputs
"this is illegal" or "this is a crime." That isn't a visual pattern a model
can learn from a photo; it depends on context, intent, and jurisdiction.
Anything claiming to do that out of the box would just be guessing and
mislabeling things with false confidence, which is worse than not doing it.

What real analytics tools do (and what this project does) is flag specific,
*explainable* behaviors — loitering, running, crowding, unattended bags —
that a human reviewer can then check. If you want *weapon detection*
specifically, that's a real, well-scoped extra step: you'd fine-tune YOLOv8
on a labeled weapons dataset (e.g. the open "weapon detection" datasets on
Roboflow Universe) and add the resulting classes to `detector.py`. I can help
you set that up if you want to take it further — it would meaningfully
strengthen the submission.

## Setup

```bash
pip install -r requirements.txt
```

First run will auto-download `yolov8n.pt` (~6MB). GPU is optional but speeds
things up a lot (`torch` will use CUDA automatically if available).

## Run

```bash
streamlit run app.py
```

This opens a browser tab. Upload a video, pick a mode in the sidebar, adjust
the confidence slider if detections look too strict/loose, and click
**Start processing**. The annotated video streams live and is also offered
as a download when finished.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI, ties everything together |
| `detector.py` | YOLOv8 wrapper — detects all object classes |
| `tracker.py` | DeepSORT wrapper — assigns persistent IDs across frames |
| `person_reid.py` | Appearance-embedding matcher for "find this person" |
| `suspicious_activity.py` | Heuristic behavior flags (loitering, running, crowding, abandoned objects) |

## Tuning notes

- `conf_threshold` (sidebar slider): lower it to catch smaller/dimmer objects
  like a ball, at the cost of more false positives.
- `person_reid.py` uses general appearance (clothing/build), not face
  recognition, so it works even from behind — but two people in near-identical
  outfits can confuse it. Swapping in a dedicated re-ID backbone like
  `torchreid`'s OSNet would make this more robust if you have time.
- `suspicious_activity.py` thresholds (`loiter_seconds`, `speed_threshold`,
  `abandon_seconds`) are tuned for a roughly 25fps webcam-ish video — adjust
  them to your footage.

## Suggested talking points for your submission

Framing the "suspicious activity" and "find a person" features as
**explainable heuristic flags** and **appearance-based re-identification**
(rather than claiming a crime-detection AI) will read as more technically
credible to reviewers — it shows you understand what's actually learnable
from a single video stream versus what would need labeled training data.
