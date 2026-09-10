# Pratique Libras 🤟

A real-time Brazilian Sign Language (Libras) alphabet recognizer, built with computer vision and machine learning. Show a sign to your camera, and the app transcribes it live on screen.

This project was inspired by a Python course I recently completed, and doubles as a practice tool for my Libras class in college.

## How it works

1. **Hand detection** — [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (Tasks API) detects both hands from a webcam feed and extracts 21 landmark points per hand.
2. **Data collection** — Landmark coordinates are recorded per sign, per hand, and saved to CSV (static signs) or sequential `.txt` files (dynamic/moving signs).
3. **Classification** — A `RandomForestClassifier` (scikit-learn) is trained on the recorded landmarks — one model per hand — to recognize which letter or number is being signed.
4. **Real-time recognition** — The trained models predict signs live from the camera feed, displaying the predicted letter/number and the model's confidence for each hand.

## Tech stack

- Python 3.14
- [OpenCV](https://opencv.org/) — camera capture and rendering
- [MediaPipe](https://ai.google.dev/edge/mediapipe) — hand landmark detection (Tasks API)
- [scikit-learn](https://scikit-learn.org/) — classification (Random Forest)
- [pandas](https://pandas.pydata.org/) — data loading and aggregation

> **Note:** This project uses MediaPipe's newer Tasks API (`mediapipe.tasks`) rather than the legacy `mediapipe.solutions` API, which was deprecated and is not guaranteed to work on recent Python versions.

## Project structure

```
pratique-libras/
├── dados/
│   ├── mao_direita/
│   │   ├── alfabeto/
│   │   │   ├── estaticos/       # one CSV per static letter (e.g. a_direita.csv)
│   │   │   └── dinamicos/       # one subfolder per dynamic letter, with sequence files
│   │   └── numeros/
│   │       ├── estaticos/
│   │       └── dinamicos/
│   └── mao_esquerda/
│       └── ...                  # same structure as mao_direita
├── models/
│   ├── hand_landmarker.task     # MediaPipe hand detection model
│   ├── libras_esquerda.pkl      # trained classifier — left hand
│   └── libras_direita.pkl       # trained classifier — right hand
├── detectar_maos_v2.py          # camera capture + landmark recording
├── train_model.py               # loads recorded data and trains the classifiers
├── reconhecer_tempo_real.py     # real-time sign recognition
└── README.md
```

## Getting started

### Requirements

- Python 3.14 (or another recent version — just make sure MediaPipe installs correctly, see note above)
- A webcam (this project was developed using [Iriun Webcam](https://iriun.com/) to use a phone as a webcam)

### Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

python -m pip install opencv-python mediapipe pandas scikit-learn
```

Download the MediaPipe hand landmarker model and place it in `models/hand_landmarker.task`:
[hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

### Recording data

```bash
python detectar_maos_v2.py
```

- Press a letter/number key to record a **static** sign — it will continuously capture landmark samples for both hands until enough samples are collected.
- For dynamic letters (`h`, `j`, `k`, `x`, `z`), press the key once to **start** recording a sequence, and press it again to **stop and save**.
- Press `ESC` to exit.

Recordings are appended (not overwritten), so you can build up your dataset gradually, across multiple sessions.

### Training the model

```bash
python train_model.py
```

This loads all recorded samples, trains one Random Forest classifier per hand, prints accuracy metrics, and saves the trained models to `models/`.

### Running real-time recognition

```bash
python reconhecer_tempo_real.py
```

Shows the camera feed with hand landmarks drawn, along with the predicted letter/number and confidence score for each detected hand. Predictions below a confidence threshold are shown as `?` to reduce false positives. Press `ESC` to exit.

## Current status & roadmap

- [x] Real-time hand landmark detection (both hands)
- [x] Static sign recognition (alphabet + numbers)
- [x] Per-hand classifiers with confidence threshold
- [ ] Robust handling of "no sign" (hand at rest) — currently a known limitation, still being studied
- [ ] Dynamic (moving) sign classification
- [ ] Full sign-to-sentence translation, going beyond isolated letters/signs

## Acknowledgments

Built as a personal learning project, with Claude (Anthropic) used as a learning aid to understand new concepts (MediaPipe, classifier design, etc.) along the way — implementation, testing, and design decisions are my own.
