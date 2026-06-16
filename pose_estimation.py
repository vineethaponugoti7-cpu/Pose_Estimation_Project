"""
Human Pose Estimation using Deep Neural Networks for Video Analysis
Name: Vineetha Ponugoti | ID: 700756986

Uses MediaPipe Tasks PoseLandmarker API (mediapipe 0.10+)
Auto-downloads the model on first run.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import time
import argparse
import os
import urllib.request

# ── Model download ─────────────────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading pose model (~6 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model downloaded.")

# ── Skeleton connections ───────────────────────────────────────────────────────
# MediaPipe Pose has 33 landmarks (same indices as before)
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),
    (24,26),(26,28),(28,30),(28,32),
]

NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer",
    "right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_pinky","right_pinky",
    "left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee",
    "left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index",
]

def side(name):
    if "left"  in name: return "left"
    if "right" in name: return "right"
    return "center"

JOINT_CLR = {"left":(0,255,255), "right":(255,0,255), "center":(0,255,0)}
LIMB_CLR  = {"left":(0,200,200), "right":(200,0,200), "center":(0,200,100)}


def draw_skeleton(frame, landmarks_list, threshold=0.5):
    h, w = frame.shape[:2]
    all_coords = []

    for person_landmarks in landmarks_list:
        coords = []
        for lm in person_landmarks:
            x   = int(lm.x * w)
            y   = int(lm.y * h)
            vis = lm.visibility if hasattr(lm, 'visibility') else 1.0
            coords.append((x, y, vis))
        all_coords.append(coords)

        for i, j in CONNECTIONS:
            if coords[i][2] > threshold and coords[j][2] > threshold:
                color = LIMB_CLR[side(NAMES[i])]
                cv2.line(frame, coords[i][:2], coords[j][:2], color, 2, cv2.LINE_AA)

        for idx, (x, y, vis) in enumerate(coords):
            if vis > threshold:
                color = JOINT_CLR[side(NAMES[idx])]
                cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 5, (255,255,255), 1, cv2.LINE_AA)

    return frame, all_coords


def compute_angles(coords, threshold=0.5):
    def angle(a, b, c):
        if any(v[2] < threshold for v in [a, b, c]):
            return None
        va = np.array([a[0]-b[0], a[1]-b[1]], float)
        vc = np.array([c[0]-b[0], c[1]-b[1]], float)
        cos = np.dot(va, vc) / (np.linalg.norm(va)*np.linalg.norm(vc)+1e-6)
        return float(np.degrees(np.arccos(np.clip(cos,-1,1))))

    return {k: v for k, v in {
        "left_elbow":  angle(coords[11], coords[13], coords[15]),
        "right_elbow": angle(coords[12], coords[14], coords[16]),
        "left_knee":   angle(coords[23], coords[25], coords[27]),
        "right_knee":  angle(coords[24], coords[26], coords[28]),
        "left_hip":    angle(coords[11], coords[23], coords[25]),
        "right_hip":   angle(coords[12], coords[24], coords[26]),
    }.items() if v is not None}


def draw_hud(frame, fps, angles, num_persons):
    h = frame.shape[0]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (260,h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def put(text, y, color=(200,200,200), scale=0.5, thick=1):
        cv2.putText(frame, text, (8,y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    put("POSE ESTIMATION", 22, (0,230,255), 0.6, 2)
    put(f"FPS     : {fps:5.1f}", 45, (0,255,120))
    put(f"Persons : {num_persons}", 65, (0,255,120))
    put("Joint Angles (deg)", 92, (180,180,180), 0.45)
    y = 110
    for name, val in angles.items():
        put(f"  {name.replace('_',' ').title():<16} {val:5.1f}", y, (200,220,200), 0.42)
        y += 18
    return frame


def run(source=0, output_path=None, show=True,
        det_conf=0.5, track_conf=0.5):

    download_model()

    RunningMode = mp.tasks.vision.RunningMode
    is_image = isinstance(source, str) and source.lower().endswith(
        (".jpg",".jpeg",".png",".bmp",".tiff"))

    running_mode = RunningMode.IMAGE if is_image else RunningMode.VIDEO

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=running_mode,
        num_poses=4,
        min_pose_detection_confidence=det_conf,
        min_pose_presence_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )

    cap    = None
    writer = None
    width  = height = 640

    if not is_image:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {source}")
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        frame_img = cv2.imread(source)
        height, width = frame_img.shape[:2]
        fps_src = 1

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        writer = cv2.VideoWriter(output_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_src, (width, height))

    prev_time   = time.time()
    frame_count = 0
    fps_display = 0.0
    timestamp_ms = 0

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:

        def process(frame):
            nonlocal prev_time, frame_count, fps_display, timestamp_ms
            frame_count += 1
            timestamp_ms += int(1000 / (fps_src if not is_image else 1))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            if is_image:
                result = landmarker.detect(mp_image)
            else:
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if frame_count % 10 == 0:
                now = time.time()
                fps_display = 10.0 / max(now - prev_time, 1e-6)
                prev_time = now

            angles      = {}
            num_persons = 0

            if result.pose_landmarks:
                num_persons = len(result.pose_landmarks)
                frame, all_coords = draw_skeleton(frame, result.pose_landmarks)
                if all_coords:
                    angles = compute_angles(all_coords[0])

            frame = draw_hud(frame, fps_display, angles, num_persons)
            return frame

        if is_image:
            out = process(frame_img)
            if show:
                cv2.imshow("Pose Estimation", out)
                cv2.waitKey(0)
            if writer:
                writer.write(out)
        else:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out = process(frame)
                if writer:
                    writer.write(out)
                if show:
                    cv2.imshow("Pose Estimation  |  Press Q to quit", out)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    if cap:    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. {frame_count} frames processed.")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose Estimation — Vineetha Ponugoti 700756986")
    parser.add_argument("--source",     default="0",  help="0=webcam or path to video/image")
    parser.add_argument("--output",     default=None, help="Save output video path")
    parser.add_argument("--no-show",    action="store_true")
    parser.add_argument("--det-conf",   type=float, default=0.5)
    parser.add_argument("--track-conf", type=float, default=0.5)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run(source, args.output, not args.no_show, args.det_conf, args.track_conf)
