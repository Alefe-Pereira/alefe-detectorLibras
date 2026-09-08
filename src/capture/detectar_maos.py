import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

INDICE_IRIUN = 0
cap = cv2.VideoCapture(INDICE_IRIUN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível ler o frame")
        break

    resultado = hands.process(frame)

    if resultado.multi_hand_landmarks:
        for mao in resultado.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, mao, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Pratique Libras - Teste", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()