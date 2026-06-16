# Human Pose Estimation for Video Analysis

Real-time human pose estimation from webcam, video, or images using deep neural networks. The application detects body landmarks, draws a colour-coded skeleton, computes joint angles, and overlays a live stats panel. Built with Python, OpenCV, and Google's MediaPipe Pose Landmarker.

This project grew out of my master's work on deep neural networks for human pose estimation in video, turned into a working, hands-on application.

## Demo

Run it on your webcam and it tracks your body in real time, drawing the skeleton and showing joint angles (elbows, knees, hips) and frame rate.

*(Add a screenshot or GIF here once you've run it — see "Adding a demo image" below.)*

## Features

- **Real-time pose tracking** from webcam, video files, or single images.
- **33 body landmarks** detected per person, drawn as a colour-coded skeleton (left side, right side, and centre joints in different colours).
- **Multi-person** support (up to 4 people in frame).
- **Joint-angle computation** — elbows, knees, and hips, calculated live from the landmark geometry.
- **On-screen stats panel** — frame rate, number of people detected, and the joint angles.
- **Save output** — write the annotated video to a file.

## How it works

The pipeline for each frame is:

1. **Capture** a frame from the webcam, video, or image (OpenCV).
2. **Convert** it to RGB and wrap it as a MediaPipe image.
3. **Detect** body landmarks with MediaPipe's Pose Landmarker, a pre-trained deep neural network.
4. **Draw** the skeleton by connecting landmark pairs, colouring limbs and joints by body side.
5. **Compute joint angles** using vector geometry — for three joints A-B-C, the angle at B is found from the dot product of vectors B→A and B→C.
6. **Overlay** a stats panel and (optionally) write the frame to an output video.

The deep learning model itself is MediaPipe's pre-trained Pose Landmarker; the application code — the capture loop, skeleton drawing, angle calculations, multi-person handling, and stats overlay — is written here.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/pose-estimation.git
cd pose-estimation
pip install -r requirements.txt
```

The pose model (~6 MB) downloads automatically the first time you run the script.

## Usage

Run on your **webcam**:

```bash
python src/pose_estimation.py --source 0
```

Run on a **video file** and save the annotated result:

```bash
python src/pose_estimation.py --source path/to/video.mp4 --output output/result.mp4
```

Run on a single **image**:

```bash
python src/pose_estimation.py --source path/to/photo.jpg
```

Press **Q** to quit the live window.

### Options

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | `0` for webcam, or a path to a video/image |
| `--output` | none | Path to save the annotated video |
| `--no-show` | off | Run without opening a display window |
| `--det-conf` | `0.5` | Minimum detection confidence |
| `--track-conf` | `0.5` | Minimum tracking confidence |

## Adding a demo image

A visual makes this repo much more engaging. Once you've run it:

1. Take a screenshot of the pose window (or save a GIF), e.g. run with `--output output/demo.mp4` and grab a frame.
2. Put the image in an `assets/` folder.
3. Replace the *Demo* section above with: `![demo](assets/demo.png)`

## Tech stack

- **Python**
- **MediaPipe** — pre-trained Pose Landmarker (deep neural network)
- **OpenCV** — video capture, drawing, and output
- **NumPy** — vector math for joint angles

## Limitations & future work

- Joint angles are computed in 2D image space, so they're approximate when limbs point toward or away from the camera. MediaPipe also outputs 3D world landmarks, which would give more accurate angles.
- Accuracy drops with heavy occlusion, motion blur, or unusual camera angles.
- Possible extensions: action or exercise recognition (e.g. counting squats from knee angles), 3D angle computation using world landmarks, and pose comparison against a reference.

## Background

This application complements a literature review I did on deep neural networks for human pose estimation in video, where I studied how CNN-based methods detect and track body keypoints. Here that idea is put into practice with a working real-time system.

