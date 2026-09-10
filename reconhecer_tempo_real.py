import cv2
import mediapipe as mp
import pickle
import os

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

CAMERA_ESPELHADA = False
NOME_EXIBICAO = {"Left": "Esquerda", "Right": "Direita"}


def corrigir_lado(nome_mao):
    if CAMERA_ESPELHADA:
        return nome_mao
    return "Right" if nome_mao == "Left" else "Left"


PASTA_MODELOS = "models"
modelo_esquerda = None
modelo_direita = None

caminho_esquerda = f"{PASTA_MODELOS}/libras_esquerda.pkl"
caminho_direita = f"{PASTA_MODELOS}/libras_direita.pkl"

if os.path.exists(caminho_esquerda):
    with open(caminho_esquerda, "rb") as f:
        modelo_esquerda = pickle.load(f)
    print("Modelo da mão esquerda carregado.")
else:
    print("Aviso: modelo da mão esquerda não encontrado.")

if os.path.exists(caminho_direita):
    with open(caminho_direita, "rb") as f:
        modelo_direita = pickle.load(f)
    print("Modelo da mão direita carregado.")
else:
    print("Aviso: modelo da mão direita não encontrado.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível ler o frame")
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    resultado = landmarker.detect_for_video(mp_image, timestamp_ms)

    if resultado.hand_landmarks:
        h, w, _ = frame.shape
        for i, mao in enumerate(resultado.hand_landmarks):
            nome_mao = corrigir_lado(resultado.handedness[i][0].category_name)

            xs = [int(ponto.x * w) for ponto in mao]
            ys = [int(ponto.y * h) for ponto in mao]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            pontos = []
            for ponto in mao:
                pontos.append(ponto.x)
                pontos.append(ponto.y)

            LIMITE_CONFIANCA = 0.78

            letra_prevista = "?"
            confianca = 0.0
            modelo = modelo_esquerda if nome_mao == "Left" else modelo_direita

            if modelo is not None:
                probabilidades = modelo.predict_proba([pontos])[0]
                indice_maior = probabilidades.argmax()
                confianca = probabilidades[indice_maior]
                classe_prevista = modelo.classes_[indice_maior]

                if confianca >= LIMITE_CONFIANCA:
                    letra_prevista = classe_prevista.upper()
                else:
                    letra_prevista = "?"

            texto_mao = f"{NOME_EXIBICAO[nome_mao]}:"
            texto_letra = f"{letra_prevista} ({confianca:.0%})"

            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

            cv2.putText(frame, texto_mao, (x_min - 20, y_min - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, texto_letra, (x_min - 20, y_min - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            for ponto in mao:
                x = int(ponto.x * w)
                y = int(ponto.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Detector de Libras - Tempo Real", frame)

    tecla_raw = cv2.waitKey(1) & 0xFF
    if tecla_raw == 27:
        break

cap.release()
cv2.destroyAllWindows()