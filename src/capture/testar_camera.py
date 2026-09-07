import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Índice {i}: câmera disponível")
        cap.release()
    else:
        print(f"Índice {i}: indisponível")