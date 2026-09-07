import cv2

INDICE_IRIUN = 0

cap = cv2.VideoCapture(INDICE_IRIUN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível ler o frame")
        break

    cv2.imshow("Pratique Libras - Teste", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()