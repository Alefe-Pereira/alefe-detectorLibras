import cv2
import mediapipe as mp
import csv
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

PASTA_DADOS = "dados"
CAMINHO_ESQUERDA = f"{PASTA_DADOS}/mao_esquerda/alfabeto.csv"
CAMINHO_DIREITA = f"{PASTA_DADOS}/mao_direita/alfabeto.csv"
PASTA_DINAMICOS = f"{PASTA_DADOS}/dinamicos"
os.makedirs(f"{PASTA_DADOS}/mao_esquerda", exist_ok=True)
os.makedirs(f"{PASTA_DADOS}/mao_direita", exist_ok=True)
os.makedirs(PASTA_DINAMICOS, exist_ok=True)

TECLAS_VALIDAS = [chr(i) for i in range(ord('a'), ord('z') + 1)] + [str(n) for n in range(10)]
LETRAS_DINAMICAS = ['h', 'j', 'k', 'x', 'z']  # ajuste com as 7 letras corretas

CAPTURAS_POR_SESSAO = 270  # padrão de capturas contínuas por letra/mão

CAMERA_ESPELHADA = False

def corrigir_lado(nome_mao):
    if CAMERA_ESPELHADA:
        return nome_mao
    return "Right" if nome_mao == "Left" else "Left"

# Estado da captura contínua (letras estáticas)
capturando_estatico = False
letra_capturando = None
contador_capturas = 0

# Estado da gravação dinâmica (sequência)
gravando_dinamico = False
letra_dinamica_atual = None
sequencia = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível ler o frame")
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    resultado = landmarker.detect_for_video(mp_image, timestamp_ms)

    maos_detectadas = {}  # nome_mao -> lista de 42 pontos

    if resultado.hand_landmarks:
        h, w, _ = frame.shape
        for i, mao in enumerate(resultado.hand_landmarks):
            nome_mao = corrigir_lado(resultado.handedness[i][0].category_name)

            xs = [int(ponto.x * w) for ponto in mao]
            ys = [int(ponto.y * h) for ponto in mao]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
            cv2.putText(frame, nome_mao, (x_min - 20, y_min - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            for ponto in mao:
                x = int(ponto.x * w)
                y = int(ponto.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            pontos_temp = []
            for ponto in mao:
                pontos_temp.append(ponto.x)
                pontos_temp.append(ponto.y)
            maos_detectadas[nome_mao] = pontos_temp

    # ===== TEXTO DE STATUS NA TELA =====
    if capturando_estatico:
        cv2.putText(frame, f"GRAVANDO: {letra_capturando} ({contador_capturas}/{CAPTURAS_POR_SESSAO})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if gravando_dinamico:
        cv2.putText(frame, f"GRAVANDO SEQUENCIA: {letra_dinamica_atual}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Pratique Libras - Teste", frame)

    tecla_raw = cv2.waitKey(1) & 0xFF
    tecla = chr(tecla_raw) if tecla_raw < 128 else ''

    if tecla_raw == 27:  # ESC
        break

    # ===== INICIAR CAPTURA (estática ou dinâmica) via tecla =====
    if tecla in TECLAS_VALIDAS and not capturando_estatico and not gravando_dinamico:
        if tecla in LETRAS_DINAMICAS:
            gravando_dinamico = True
            letra_dinamica_atual = tecla
            sequencia = []
            print(f"Gravação de sequência iniciada: {tecla}")
        else:
            capturando_estatico = True
            letra_capturando = tecla
            contador_capturas = 0
            print(f"Captura contínua iniciada: {tecla}")

    # Permite parar a gravação dinâmica apertando a mesma tecla de novo
    elif tecla in LETRAS_DINAMICAS and gravando_dinamico and tecla == letra_dinamica_atual:
        gravando_dinamico = False
        print(f"Gravação parada. {len(sequencia)} frames capturados.")

        pasta_letra = f"{PASTA_DINAMICOS}/{letra_dinamica_atual}"
        os.makedirs(pasta_letra, exist_ok=True)
        caminho = f"{pasta_letra}/{letra_dinamica_atual}_{len(os.listdir(pasta_letra))}.txt"
        with open(caminho, "w") as arquivo:
            for frame_pontos in sequencia:
                linha = ",".join(str(n) for n in frame_pontos)
                arquivo.write(linha + "\n")
        print(f"Salvo em {caminho}")
        letra_dinamica_atual = None

    # ===== ACUMULA SEQUENCIA DINAMICA =====
    if gravando_dinamico:
        pontos_esquerda = maos_detectadas.get("Left", [0] * 42)
        pontos_direita = maos_detectadas.get("Right", [0] * 42)
        sequencia.append(pontos_esquerda + pontos_direita)

    # ===== CAPTURA CONTINUA ESTATICA =====
    if capturando_estatico:
        # salva um frame por iteração do loop, para cada mão detectada
        for nome_mao, pontos in maos_detectadas.items():
            linha = [letra_capturando] + pontos
            caminho_csv = CAMINHO_ESQUERDA if nome_mao == "Left" else CAMINHO_DIREITA
            with open(caminho_csv, "a", newline="") as arquivo:
                escritor = csv.writer(arquivo)
                escritor.writerow(linha)

        if maos_detectadas:
            contador_capturas += 1

        if contador_capturas >= CAPTURAS_POR_SESSAO:
            capturando_estatico = False
            print(f"Captura de '{letra_capturando}' concluída ({contador_capturas} amostras).")
            letra_capturando = None
            contador_capturas = 0

cap.release()
cv2.destroyAllWindows()