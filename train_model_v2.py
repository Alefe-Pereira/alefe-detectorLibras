import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

PASTA_DADOS = "dados"
PASTA_MODELOS = "models"
os.makedirs(PASTA_MODELOS, exist_ok=True)

COLUNAS_PONTOS = [f"p{i}" for i in range(42)]


def carregar_dados(nome_mao):
    pastas_estaticos = [
        f"{PASTA_DADOS}/mao_{nome_mao}/alfabeto/estaticos",
        f"{PASTA_DADOS}/mao_{nome_mao}/numeros/estaticos",
    ]

    tabelas = []

    for pasta in pastas_estaticos:
        if not os.path.exists(pasta):
            continue

        for caminho_arquivo in glob.glob(f"{pasta}/*.csv"):
            nome_arquivo = os.path.basename(caminho_arquivo)
            letra = nome_arquivo.split(f"_{nome_mao}")[0]  # "a_direita.csv" -> "a"

            df_letra = pd.read_csv(caminho_arquivo, header=None, names=COLUNAS_PONTOS)
            df_letra.insert(0, "letra", letra)
            tabelas.append(df_letra)

    if not tabelas:
        return pd.DataFrame()

    return pd.concat(tabelas, ignore_index=True)


def treinar_e_avaliar(df, nome_mao):
    if df.empty:
        print(f"Nenhum dado encontrado para a mão {nome_mao}.")
        return

    print(f"\n===== Mão {nome_mao} =====")
    print(f"Total de amostras: {len(df)}")
    print("Amostras por letra/número:")
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
    df_esquerda = carregar_dados("esquerda")
    df_direita = carregar_dados("direita")

    treinar_e_avaliar(df_esquerda, "esquerda")
    treinar_e_avaliar(df_direita, "direita")