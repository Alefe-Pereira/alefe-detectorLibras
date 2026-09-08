import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)
landmarker = HandLandmarker.create_from_options(options)

INDICE_IRIUN = 0
cap = cv2.VideoCapture(INDICE_IRIUN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível ler o frame")
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    resultado = landmarker.detect_for_video(mp_image, timestamp_ms)

    if resultado.hand_landmarks:
        for mao in resultado.hand_landmarks:
            for ponto in mao:
                x = int(ponto.x * frame.shape[1])
                y = int(ponto.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Pratique Libras - Teste", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()