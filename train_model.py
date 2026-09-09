import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

PASTA_DADOS = "dados"
CAMINHO_ESQUERDA = f"{PASTA_DADOS}/mao_esquerda/alfabeto.csv"
CAMINHO_DIREITA = f"{PASTA_DADOS}/mao_direita/alfabeto.csv"
PASTA_MODELOS = "models"
os.makedirs(PASTA_MODELOS, exist_ok=True)


def carregar_dados(caminho):
    if not os.path.exists(caminho):
        return pd.DataFrame()
    # primeira coluna = letra, demais = 42 pontos (x,y de 21 landmarks)
    colunas = ["letra"] + [f"p{i}" for i in range(42)]
    df = pd.read_csv(caminho, header=None, names=colunas)
    return df


def treinar_e_avaliar(df, nome_mao):
    if df.empty:
        print(f"Nenhum dado encontrado para a mão {nome_mao}.")
        return

    print(f"\n===== Mão {nome_mao} =====")
    print(f"Total de amostras: {len(df)}")
    print("Amostras por letra:")
    print(df["letra"].value_counts())

    X = df.drop("letra", axis=1)
    y = df["letra"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    modelo.fit(X_treino, y_treino)

    y_pred = modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, y_pred)

    print(f"\nAcurácia no conjunto de TESTE (dados nunca vistos): {acuracia:.2%}")
    print("\nRelatório de classificação:")
    print(classification_report(y_teste, y_pred))

    caminho_modelo = f"{PASTA_MODELOS}/libras_{nome_mao}.pkl"
    with open(caminho_modelo, "wb") as f:
        pickle.dump(modelo, f)
    print(f"Modelo salvo em {caminho_modelo}")


if __name__ == "__main__":
    df_esquerda = carregar_dados(CAMINHO_ESQUERDA)
    df_direita = carregar_dados(CAMINHO_DIREITA)

    treinar_e_avaliar(df_esquerda, "esquerda")
    treinar_e_avaliar(df_direita, "direita")