
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script unificado: Treinamento de RNA + Extrapolação de RSRP + Relatórios + Visualizações.
Salva resultados completos (CSV, PNG, TXT) na pasta /home/drcoliveira/UFU/Doutorado/Defesa/Dataset/RNA/PCI_xxx
"""

import os
import json
import time
import datetime as dt
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.initializers import GlorotNormal
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ===== Energia & Carbono (parâmetros simples, configuráveis) =====
# Intensidades médias de carbono por kWh (kgCO2e/kWh) — valores típicos aproximados.
# Ajustar conforme fonte/dataset de emissões.
FATORES_CO2_KG_POR_KWH = {
    "Brasil": 0.1295,        # ~0,1295 kgCO2e/kWh --> CLIMATE TRANSPARENCY REPORT: COMPARING G20 CLIMATE ACTION
    "G20": 0.4447,           # ~0,4447 kgCO2e/kWh --> CLIMATE TRANSPARENCY REPORT: COMPARING G20 CLIMATE ACTION
    "União Europeia": 0.30,  # ~0.30 kgCO2e/kWh (média)
    "Ásia": 0.60,            # ~0.60 kgCO2e/kWh (média)
    "EUA": 0.40,             # ~0.40 kgCO2e/kWh (média)
}

def estimar_kwh_e_co2(tempo_segundos: float, potencia_watts: float) -> tuple[float, dict]:
    """
    Retorna (kwh, dict_regiao->kgCO2e) para o tempo e potência informados.
    kWh = (W * s) / 3.6e6
    """
    kwh = (potencia_watts * tempo_segundos) / 3_600_000.0
    co2 = {reg: kwh * fator for reg, fator in FATORES_CO2_KG_POR_KWH.items()}
    return kwh, co2

# Hiperparâmetros
MAX_EPOCAS = 50000
N_NOS = 20
ATIVACAO = "relu"
LR = 1e-4
BATCH_SIZE = 4
TEST_SIZE = 0.10
VAL_RATIO_DENTRO_TREINO = 0.10

class EarlyStoppingByLossValue(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=1e-5, verbose=1):
        super().__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.stopped_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        atual = logs.get(self.monitor)
        if atual is not None and atual < self.value:
            if self.verbose:
                print(f"Época {epoch+1:05d}: early stopping - {self.monitor}: {atual:.6g}")
            self.stopped_epoch = epoch + 1
            self.model.stop_training = True


# Funções da rede
def construir_modelo() -> tf.keras.Model:
    model = Sequential(name="MLP_RSRP_norm")
    init = GlorotNormal(seed=42)
    for i in range(4):
        model.add(Dense(N_NOS, activation=ATIVACAO, kernel_initializer=init, input_dim=2 if i == 0 else None))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss="mean_squared_error")
    return model

def salvar_pesos_json(model: tf.keras.Model, path: Path):
    dump = {}
    for layer in model.layers:
        wts = layer.get_weights()
        if not wts:
            continue
        W, b = wts
        dump[layer.name] = {
            "W": W.tolist(),
            "b": b.tolist()
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2)

def carregar_pesos_json(path: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers = []
    for k in sorted(data.keys()):
        W = np.array(data[k]["W"], dtype=np.float32)
        b = np.array(data[k]["b"], dtype=np.float32)
        layers.append((W, b))
    return layers

def forward_predict(X: np.ndarray, layers: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    H = X
    for i, (W, b) in enumerate(layers):
        H = H @ W + b
        if i < len(layers) - 1:
            H = np.maximum(H, 0.0)
    return H.squeeze()

def norm_to_dbm(v: np.ndarray) -> np.ndarray:
    return np.clip(v, 0.0, 1.0) * 70.0 - 120.0

def plot_coverage(df: pd.DataFrame, value_col: str, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    sc = ax.scatter(
        df["lon"], df["lat"],
        c=df[value_col], cmap="plasma", s=6, vmin=-120, vmax=-50
    )
    plt.colorbar(sc, ax=ax, shrink=0.9).set_label("RSRP (dBm)")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle='--', alpha=0.4)
    try:
        ax.set_aspect("equal", adjustable="datalim")
    except:
        pass
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close()


def main():
    pci = input("Digite o PCI (ex: 009): ").strip().zfill(3)
    base_dir = Path("/home/drcoliveira/UFU/Doutorado/Defesa/Dataset")
    # pasta_rna = base_dir / "RNA" / "Leitura"  # Leitura2 tem os dados com aumento somente na A_A
    pasta_rna = base_dir / "RNA" / "Leitura2" # Leitura2 tem os dados com aumento não somente na A_A, mas em A_T
    pasta_saida = base_dir / "RNA" / f"PCI_{pci}"
    pasta_saida.mkdir(parents=True, exist_ok=True)

    entrada_csv = sorted(pasta_rna.glob(f"PCI_{pci}_base*.csv"))[0]
    df = pd.read_csv(entrada_csv)
    X = df[["lat_norm", "long_norm"]].to_numpy()
    y = df["RSRP_norm"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_RATIO_DENTRO_TREINO, random_state=42)

    model = construir_modelo()

    # === Converter alvo de RMSE (4 dB) para MSE no espaço normalizado ===
    RMSE_DB_ALVO = 4.0
    mse_limite = (RMSE_DB_ALVO / 70.0) ** 2

    # Early stopping por valor absoluto (RMSE < 4 dB → val_loss < mse_limite)
    early_rmse = EarlyStoppingByLossValue(
        monitor='val_loss',
        value=mse_limite,
        verbose=1
    )

    # Early stopping por platô em val_loss
    early_plateau = EarlyStopping(
        monitor='val_loss',
        patience=300,      # originalmente em 200, ajustar para mais ou menos tolerância
        min_delta=1e-4,    # melhora mínima para contar como “melhorou”
        restore_best_weights=True,
        verbose=1
    )

    t0 = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCAS,
        batch_size=BATCH_SIZE,
        verbose=0
        # callbacks=[early_rmse, early_plateau]   # <<< usa os dois
    )
    tempo_s = time.perf_counter() - t0

    # ===== Estimativa Energia & Carbono =====
    # Detecta GPU para definir potência padrão; pode sobrescrever com POWER_WATTS
    tem_gpu = len(tf.config.list_physical_devices('GPU')) > 0
    potencia_default_w = 125.0 if tem_gpu else 45.0
    potencia_w = float(os.getenv("POWER_WATTS", potencia_default_w))

    kwh, co2 = estimar_kwh_e_co2(tempo_s, potencia_w)

    # Avaliação
    y_pred = model.predict(X_test).flatten()
    y_test_db = norm_to_dbm(y_test)
    y_pred_db = norm_to_dbm(y_pred)
    erros_db = y_pred_db - y_test_db
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse_db = np.sqrt(mse) * 70
    media_erro_db = np.mean(erros_db)
    desvio_erro_db = np.std(erros_db)

    # Gráficos
    hist_png = pasta_saida / f"hist_erro_PCI_{pci}_dB.png"
    disp_png = pasta_saida / f"dispersao_PCI_{pci}_dB.png"

    plt.figure()
    plt.hist(erros_db, bins=30, edgecolor='black')
    plt.title("Distribuição dos Erros (em dB)")
    plt.xlabel("Erro (dB)")
    plt.ylabel("Frequência")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(hist_png, dpi=300)
    plt.close()

    plt.figure()
    plt.scatter(y_test_db, y_pred_db, s=10, alpha=0.6)
    plt.plot([-120, -50], [-120, -50], 'r--')
    plt.xlabel("RSRP Medido (dBm)")
    plt.ylabel("RSRP Previsto (dBm)")
    plt.title("Dispersão: RSRP Medido vs Previsto")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(disp_png, dpi=300)
    plt.close()

    # Salvar pesos
    pesos_path = pasta_saida / f"weights_PCI_{pci}.json"
    salvar_pesos_json(model, pesos_path)

    # TXT

    # Descobrir qual critério parou o treinamento
    epoca_rmse = early_rmse.stopped_epoch
    epoca_plateau = early_plateau.stopped_epoch

    if epoca_rmse is not None:
        epoca_parada = epoca_rmse
        motivo_parada = "RMSE < 4 dB (via val_loss)"
    elif epoca_plateau is not None:
        epoca_parada = epoca_plateau
        motivo_parada = "Platô em val_loss (EarlyStopping)"
    else:
        epoca_parada = MAX_EPOCAS
        motivo_parada = "Limite máximo de épocas atingido"


    relatorio_path = pasta_saida / f"resultado_PCI_{pci}_RNA.txt"
    with open(relatorio_path, "w", encoding="utf-8") as f:
        f.write(f"Arquivo de entrada: {entrada_csv}\n")
        f.write(f"1) Épocas até convergência: {epoca_parada} ({motivo_parada})\n")
        f.write(f"2) MSE: {mse:.8f}\n")
        f.write(f"3) R²: {r2:.8f}\n")
        f.write(f"   (Equivalente em escala dB: RMSE = {rmse_db:.3f} dB)\n")
        f.write(f"   Erro médio = {media_erro_db:.3f} dB | Desvio-padrão = {desvio_erro_db:.3f} dB\n")
        f.write(f"4) Tempo de execução: {str(dt.timedelta(seconds=int(tempo_s)))} ({tempo_s:.3f} s)\n")
        f.write("\n--- Energia & Carbono (estimativas) ---\n")
        f.write(f"Potência assumida: {potencia_w:.1f} W  |  Ajustável via env POWER_WATTS\n")
        f.write(f"Consumo estimado: {kwh:.6f} kWh\n")
        f.write("Pegada de carbono (kgCO2e):\n")
        for reg, kg in co2.items():
            f.write(f"  - {reg}: {kg:.6f}\n")

        f.write(f"Pesos salvos em: {pesos_path}\n")

    # Extrapolação
    matriz_csv = base_dir / "RNA" / "matriz_analise.csv"
    df_matriz = pd.read_csv(matriz_csv)
    X_ext = df_matriz[["lat_norm", "lon_norm"]].to_numpy()
    layers = carregar_pesos_json(pesos_path)
    y_ext = forward_predict(X_ext, layers)
    y_ext_db = norm_to_dbm(y_ext)

    df_out = df_matriz.copy()
    df_out["rsrp_norm"] = y_ext
    df_out["rsrp_dBm"] = y_ext_db

    out_csv = pasta_saida / f"PCI_{pci}_extrapolado_RNA.csv"
    out_png = pasta_saida / f"PCI_{pci}_extrapolado_RNA.png"
    df_out.to_csv(out_csv, index=False)
    plot_coverage(df_out, "rsrp_dBm", out_png, f"Mapa de Cobertura – PCI {pci}")

    print(f"[OK] Relatório: {relatorio_path}")
    print(f"[OK] CSV extrapolado: {out_csv}")
    print(f"[OK] PNG gerado: {out_png}")
    print(f"[OK] Pesos em JSON: {pesos_path}")
    print(f"[INFO] MSE={mse:.6f} | R²={r2:.6f} | RMSE={rmse_db:.3f} dB | Erro médio={media_erro_db:.3f} dB")
    print(f"[INFO] Potência assumida: {potencia_w:.1f} W | Consumo: {kwh:.6f} kWh")
    for reg, kg in co2.items():
        print(f"[INFO] CO2 {reg}: {kg:.6f} kgCO2e")

if __name__ == "__main__":
    main()
