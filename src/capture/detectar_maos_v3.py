NOME_EXIBICAO = {"Left": "Esquerda", "Right": "Direita"}
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
NUMEROS = [str(n) for n in range(10)]


def obter_categoria(tecla):
    return "numeros" if tecla in NUMEROS else "alfabeto"


def caminho_csv_estatico(tecla, nome_mao):
    categoria = obter_categoria(tecla)
    pasta_mao = "mao_direita" if nome_mao == "Right" else "mao_esquerda"
    pasta = f"{PASTA_DADOS}/{pasta_mao}/{categoria}/estaticos"
    os.makedirs(pasta, exist_ok=True)
    sufixo_mao = "direita" if nome_mao == "Right" else "esquerda"
    return f"{pasta}/{tecla}_{sufixo_mao}.csv"


def pasta_dinamico(tecla, nome_mao):
    categoria = obter_categoria(tecla)
    pasta_mao = "mao_direita" if nome_mao == "Right" else "mao_esquerda"
    pasta = f"{PASTA_DADOS}/{pasta_mao}/{categoria}/dinamicos/{tecla}"
    os.makedirs(pasta, exist_ok=True)
    return pasta


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
            cv2.putText(frame, NOME_EXIBICAO[nome_mao], (x_min - 20, y_min - 30),
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

    elif tecla in LETRAS_DINAMICAS and gravando_dinamico and tecla == letra_dinamica_atual:
        gravando_dinamico = False
        print(f"Gravação parada. {len(sequencia)} frames capturados.")

        pasta_esquerda = pasta_dinamico(letra_dinamica_atual, "Left")
        pasta_direita = pasta_dinamico(letra_dinamica_atual, "Right")

        caminho_esquerda = f"{pasta_esquerda}/{letra_dinamica_atual}_{len(os.listdir(pasta_esquerda))}.txt"
        caminho_direita = f"{pasta_direita}/{letra_dinamica_atual}_{len(os.listdir(pasta_direita))}.txt"

        with open(caminho_esquerda, "w") as arq_esq, open(caminho_direita, "w") as arq_dir:
            for frame_pontos in sequencia:
                pontos_esq = frame_pontos[:42]
                pontos_dir = frame_pontos[42:]
                arq_esq.write(",".join(str(n) for n in pontos_esq) + "\n")
                arq_dir.write(",".join(str(n) for n in pontos_dir) + "\n")

        print(f"Salvo em {caminho_esquerda} e {caminho_direita}")
        letra_dinamica_atual = None

    if gravando_dinamico:
        pontos_esquerda = maos_detectadas.get("Left", [0] * 42)
        pontos_direita = maos_detectadas.get("Right", [0] * 42)
        sequencia.append(pontos_esquerda + pontos_direita)

    if capturando_estatico:
        for nome_mao, pontos in maos_detectadas.items():
            caminho_csv = caminho_csv_estatico(letra_capturando, nome_mao)
            with open(caminho_csv, "a", newline="") as arquivo:
                escritor = csv.writer(arquivo)
                escritor.writerow(pontos)

        if maos_detectadas:
            contador_capturas += 1

        if contador_capturas >= CAPTURAS_POR_SESSAO:
            capturando_estatico = False
            print(f"Captura de '{letra_capturando}' concluída ({contador_capturas} amostras).")
            letra_capturando = None
            contador_capturas = 0

cap.release()
cv2.destroyAllWindows()