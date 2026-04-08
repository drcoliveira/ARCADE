#!/usr/bin/env python3

import os
import time
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

try:
    import cma
except ImportError:
    raise ImportError("Instale a biblioteca 'cma' com: pip install cma")

CSV_PATH = "/home/drcoliveira/UFU/Doutorado/Defesa/Dataset/6_Merging/Mapa_Composicao_k075.csv"
LOG_DIR  = "/home/drcoliveira/UFU/Doutorado/Defesa/Dataset/8_CMAES/NovaCalibracao"
os.makedirs(LOG_DIR, exist_ok=True)

# ===== Robustez (multi-seed) =====
N_SEEDS = 1  # default 20
SEED_START = 177
SAVE_PER_SEED_LOGS = True  # se False, salva só o robusto final

BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 123
# Lista de colunas de PCI na ordem dos elementos de P
PCI_COLS = [
    "PCI_009", "PCI_010", "PCI_011",
    "PCI_165", "PCI_166", "PCI_167",
    "PCI_243", "PCI_244", "PCI_245", 
    "PCI_246", "PCI_247", "PCI_248",
    "PCI_252", "PCI_253", "PCI_254",
    "PCI_333", "PCI_334", "PCI_335",
    "PCI_395", "PCI_396", "PCI_397",
    "PCI_404", "PCI_405", "PCI_406",
]
K = len(PCI_COLS)

# Parâmetros do funcional
w_cov = 1.0 # Padrão: 1.0
w_dom = 3.5 # Melhor encontrado: 3.5
w_SDI = 1.0 # Melhor encontrado: 1.0
lambda_reg = 0.1 # Melhor encontrado: 0.1

S_min = -105.0
S_max = -75.0

Delta_min = 0.0
Delta_ref = 9.0

sigma_P = 4.0 # Melhor: 4.0
RSRP_min_SDI = -110.0

DELTA_VALUES = np.array([-9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0], dtype=float)

# CMA-ES
SIGMA0 = 3.0
BOUNDS = (-9.0, 9.0)
MAXITER = 100 # Melhor: 100

# ------------------------------------------------------------
# 2. Leitura do CSV e pré-processamento
# ------------------------------------------------------------

def load_data(csv_path: str):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")
    df = pd.read_csv(csv_path)

    for col in ["latitude", "longitude"]:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente no CSV: {col}")
    for col in PCI_COLS:
        if col not in df.columns:
            raise ValueError(f"Coluna PCI ausente no CSV: {col}")

    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)
    rsrp = df[PCI_COLS].to_numpy(dtype=float)  # N x K
    return lat, lon, rsrp

def latlon_to_xy(lat_deg: np.ndarray, lon_deg: np.ndarray):
    R = 6371000.0
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    lat0 = np.mean(lat_rad)
    lon0 = np.mean(lon_rad)
    x = R * (lon_rad - lon0) * np.cos(lat0)
    y = R * (lat_rad - lat0)
    return x, y

# ------------------------------------------------------------
# 3. Cálculo de SDI
# ------------------------------------------------------------

def compute_SDI_for_P(rsrp_mat: np.ndarray, x_m: np.ndarray, y_m: np.ndarray,
                      deltaP: np.ndarray, rsrp_min: float = RSRP_min_SDI):
    N, K_local = rsrp_mat.shape
    assert K_local == len(deltaP)

    rsrp_prime = rsrp_mat + deltaP.reshape(1, -1)
    SDI = np.zeros(K_local, dtype=float)

    for k in range(K_local):
        rsrp_k = rsrp_prime[:, k]
        mask = rsrp_k >= rsrp_min
        if not np.any(mask):
            SDI[k] = 0.0
            continue

        rsrp_k_sel = rsrp_k[mask]
        x_sel = x_m[mask]
        y_sel = y_m[mask]

        w = (rsrp_k_sel + 120.0) / 70.0
        w = np.clip(w, 0.0, 1.0)
        w_sum = w.sum()
        if w_sum <= 0:
            SDI[k] = 0.0
            continue

        x_bar = np.sum(w * x_sel) / w_sum
        y_bar = np.sum(w * y_sel) / w_sum

        r = np.sqrt((x_sel - x_bar) ** 2 + (y_sel - y_bar) ** 2)
        SDI[k] = np.sum(w * r) / w_sum

    mu_SDI = np.mean(SDI)
    V_SDI = np.mean((SDI - mu_SDI) ** 2)
    return SDI, mu_SDI, V_SDI

# ------------------------------------------------------------
# 4. Cálculo de C_i(P), D_i(P), J_base e J_til(P)
# ------------------------------------------------------------

def compute_C_and_D(rsrp_mat: np.ndarray, deltaP: np.ndarray):
    N, K_local = rsrp_mat.shape
    rsrp_prime = rsrp_mat + deltaP.reshape(1, -1)

    S = np.max(rsrp_prime, axis=1)
    k_star = np.argmax(rsrp_prime, axis=1)

    rsrp_for_I = rsrp_prime.copy()
    rows = np.arange(N)
    rsrp_for_I[rows, k_star] = -1e9
    I = np.max(rsrp_for_I, axis=1)

    Delta = S - I

    C = np.zeros(N, dtype=float)
    mask_mid = (S > S_min) & (S < S_max)
    C[mask_mid] = (S[mask_mid] - S_min) / (S_max - S_min)
    C[S >= S_max] = 1.0

    D = np.zeros(N, dtype=float)
    mask_cov = S > S_min
    mask_mid_D = mask_cov & (Delta > Delta_min) & (Delta < Delta_ref)
    D[mask_mid_D] = (Delta[mask_mid_D] - Delta_min) / (Delta_ref - Delta_min)
    mask_high_D = mask_cov & (Delta >= Delta_ref)
    D[mask_high_D] = 1.0

    return C, D

def compute_Jtil(rsrp_mat: np.ndarray,
                 x_m: np.ndarray,
                 y_m: np.ndarray,
                 deltaP_cont: np.ndarray,
                 sigma_SDI: float):

    # USAR DIRETAMENTE O VETOR CONTÍNUO
    deltaP = np.array(deltaP_cont, dtype=float)

    # C_i e D_i
    C, D = compute_C_and_D(rsrp_mat, deltaP)
    N = len(C)

    J_base = np.mean(w_cov * C + w_dom * D)

    # Regularização (agora contínua)
    R_P = np.mean((deltaP / sigma_P) ** 2)

    # SDI
    SDI, mu_SDI, V_SDI = compute_SDI_for_P(
        rsrp_mat, x_m, y_m, deltaP, rsrp_min=RSRP_min_SDI
    )

    J_til = J_base - lambda_reg * R_P - w_SDI * V_SDI / (sigma_SDI ** 2)

    return J_til, deltaP, J_base, R_P, SDI, mu_SDI, V_SDI


def extract_best_continuous_from_robust_log(log_path: str, pci_cols=PCI_COLS) -> np.ndarray:
    """
    Lê o arquivo CMAES_robusto_log_XXX.txt e extrai o vetor
    'Melhor solução contínua (antes da quantização)'.

    Retorna np.ndarray shape (K,).
    """
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Log não encontrado: {log_path}")

    with open(log_path, "r", encoding="utf-8") as f:
        txt = f.read()

    # pega o bloco após o marcador
    marker = "Melhor solução contínua (antes da quantização):"
    if marker not in txt:
        raise ValueError(f"Seção não encontrada no log: '{marker}'")

    tail = txt.split(marker, 1)[1]

    # captura linhas do tipo: PCI_009: -0.252 dB
    found = dict(re.findall(r"(PCI_\d{3})\s*:\s*([-+]?\d+(?:\.\d+)?)\s*dB", tail))

    missing = [p for p in pci_cols if p not in found]
    if missing:
        raise ValueError(f"PCIs faltando no log: {missing}")

    delta = np.array([float(found[p]) for p in pci_cols], dtype=float)
    return delta


def compute_S_I_delta_from_continuous(rsrp_mat: np.ndarray, deltaP_cont: np.ndarray):
    """
    Aplica deltaP_cont diretamente (contínuo) à matriz rsrp (N x K),
    e retorna:
      S: maior RSRP por ponto (N,)
      I: segundo maior RSRP por ponto (N,)
      Delta: S - I (N,)
    """
    rsrp_prime = rsrp_mat + deltaP_cont.reshape(1, -1)

    # maior
    S = np.max(rsrp_prime, axis=1)

    # segundo maior: particiona para pegar top-2 sem ordenar tudo
    # top2 será (N,2) com os 2 maiores (não garantido ordenado),
    # então S2 = menor desses 2
    top2 = np.partition(rsrp_prime, kth=rsrp_prime.shape[1]-2, axis=1)[:, -2:]
    I = np.min(top2, axis=1)

    Delta = S - I
    return S, I, Delta

def compute_RSRP_max(rsrp_mat: np.ndarray, deltaP_cont: np.ndarray):
    """
    Retorna o RSRP máximo por ponto da grade após aplicar deltaP_cont.
    """
    rsrp_prime = rsrp_mat + deltaP_cont.reshape(1, -1)
    RSRP_max = np.max(rsrp_prime, axis=1)
    return RSRP_max

def export_coverage_csv(lat, lon, RSRP_max, out_csv):
    df = pd.DataFrame({
        "latitude": lat,
        "longitude": lon,
        "RSRP_max": RSRP_max
    })
    df.to_csv(out_csv, index=False)


def build_regular_grid(lat: np.ndarray, lon: np.ndarray):
    """
    Reconstrói uma malha regular a partir de vetores lat/lon 1D.
    Retorna:
      Lon_grid, Lat_grid, shape (Ny, Nx)
      idx_map: índices para reorganizar vetores -> grid
    """
    lat_vals = np.unique(lat)
    lon_vals = np.unique(lon)

    lat_vals.sort()
    lon_vals.sort()

    Ny = len(lat_vals)
    Nx = len(lon_vals)

    # mapa (lat,lon) -> índice no grid
    lat_to_i = {v: i for i, v in enumerate(lat_vals)}
    lon_to_j = {v: j for j, v in enumerate(lon_vals)}

    idx_map = np.empty((Ny, Nx), dtype=int)
    for n in range(len(lat)):
        i = lat_to_i[lat[n]]
        j = lon_to_j[lon[n]]
        idx_map[i, j] = n

    Lon, Lat = np.meshgrid(lon_vals, lat_vals)
    return Lon, Lat, idx_map


def plot_coverage_map(lat: np.ndarray, lon: np.ndarray, rsrp_mat: np.ndarray,
                      deltaP_cont: np.ndarray, out_path: str, title: str):
    """
    Mapa de cobertura com células preenchidas (grid).
    0 = branco (sem cobertura)
    1 = vermelho (com interferência)
    2 = azul (sem interferência)
    """
    S, I, Delta = compute_S_I_delta_from_continuous(rsrp_mat, deltaP_cont)

    # classes
    cls = np.zeros(len(S), dtype=int)
    cls[(S > S_min) & (Delta < Delta_ref)] = 1
    cls[(S > S_min) & (Delta >= Delta_ref)] = 2

    n_white = int((cls == 0).sum())
    n_red   = int((cls == 1).sum())
    n_blue  = int((cls == 2).sum())
    n_total = len(cls)

    # --- monta grid ---
    Lon, Lat, idx_map = build_regular_grid(lat, lon)
    cls_grid = cls[idx_map]

    # colormap discreto
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["white", "red", "blue"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    plt.figure(figsize=(9, 8))
    plt.pcolormesh(Lon, Lat, cls_grid, cmap=cmap, norm=norm, shading="nearest")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.grid(True, linewidth=0.3, alpha=0.4)

    # legenda manual (porque pcolormesh não gera automática)
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color="white", label="Sem cobertura"),
        mpatches.Patch(color="red",   label="Cobertura c/ interferência"),
        mpatches.Patch(color="blue",  label="Cobertura s/ interferência"),
    ]
    plt.legend(handles=legend_patches, loc="best")

    # texto com contagens
    txt = (f"Pontos: branco={n_white} | vermelho={n_red} | "
           f"azul={n_blue} | total={n_total}")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.figtext(0.5, 0.01, txt, ha="center", va="bottom", fontsize=10)

    plt.savefig(out_path, dpi=300)
    plt.close()

    return {
        "sem_cobertura": n_white,
        "com_interferencia": n_red,
        "sem_interferencia": n_blue,
        "total": n_total
    }

def plot_rsrp_map(lat, lon, RSRP_max, out_path, title):
    Lon, Lat, idx_map = build_regular_grid(lat, lon)
    grid = RSRP_max[idx_map]

    cmap, norm, bounds = get_pastel_rsrp_colormap()

    plt.figure(figsize=(9, 8))
    plt.pcolormesh(Lon, Lat, grid, cmap=cmap, norm=norm, shading="nearest")
    plt.colorbar(label="RSRP máximo (dBm)", ticks=bounds)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def get_pastel_rsrp_colormap():
    bounds = [-120, -100, -95, -90, -85, -80, -75, 100]
    colors = [
        "#ffffff",  # -120 a -100
        "#fff2a6",  # -100 a -95
        "#ffe680",  # -95 a -90
        "#ffb366",  # -90 a -85
        "#ff8c8c",  # -85 a -80
        "#ff6666",  # -80 a -75
        "#cc3333",  # > -75
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds

def plot_rsrp_difference_map(lat, lon, RSRP_orig, RSRP_corr, out_path, title):
    """
    Branco: RSRP_corr < -105 dBm (prioridade máxima)
    Verde: RSRP aumentou
    Laranja: RSRP diminuiu
    Azul: sem alteração (< 0.1 dB)
    """
    diff = RSRP_corr - RSRP_orig

    THR_DIFF = 0.1   # dB — limiar de indiferença
    THR_WHITE = -105 # dBm — corte absoluto para ficar branco

    # classes:
    # -2 = branco (RSRP_corr < -105)
    # -1 = laranja (piorou)
    #  0 = azul (sem alteração)
    # +1 = verde (melhorou)
    cls = np.zeros(len(diff), dtype=int)

    cls[diff >= THR_DIFF] = 1
    cls[diff <= -THR_DIFF] = -1
    # |diff| < THR_DIFF fica 0 (azul)

    # OVERRIDE: abaixo de -105 dBm fica branco, independente do diff
    cls[RSRP_corr < THR_WHITE] = -2

    Lon, Lat, idx_map = build_regular_grid(lat, lon)
    grid = cls[idx_map]

    cmap = ListedColormap(["white", "orange", "blue", "green"])
    norm = BoundaryNorm([-2.5, -1.5, -0.5, 0.5, 1.5], cmap.N)

    plt.figure(figsize=(9, 8))
    plt.pcolormesh(Lon, Lat, grid, cmap=cmap, norm=norm, shading="nearest")

    import matplotlib.patches as mpatches
    legend = [
        mpatches.Patch(color="white",  label="RSRP < -105 dBm"),
        mpatches.Patch(color="green",  label="RSRP aumentou"),
        mpatches.Patch(color="orange", label="RSRP diminuiu"),
        mpatches.Patch(color="blue",   label="Sem alteração"),
    ]
    plt.legend(handles=legend, loc="best")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ------------------------------------------------------------
# 6. Bootstrap para IC (robustez)
# ------------------------------------------------------------

def bootstrap_ci_mean(x: np.ndarray, B: int = BOOTSTRAP_B, alpha: float = 0.05, seed: int = BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    boots = np.empty(B, dtype=float)
    for b in range(B):
        boots[b] = np.mean(rng.choice(x, size=n, replace=True))
    lo = np.quantile(boots, alpha/2)
    hi = np.quantile(boots, 1 - alpha/2)
    return float(lo), float(hi)

# ------------------------------------------------------------
# 7. Execução robusta (multi-seed) + seleção formal
# ------------------------------------------------------------

def find_next_idx(log_dir: str) -> int:
    idx = 1
    while True:
        path = os.path.join(log_dir, f"CMAES_robusto_log_{idx:03d}.txt")
        if not os.path.exists(path):
            return idx
        idx += 1

def write_single_log(path: str, seed: int, csv_path: str, rsrp_mat: np.ndarray,
                     sigma_SDI: float, best_cont: np.ndarray, best_P: np.ndarray,
                     J_til_best: float, J_base_best: float, R_best: float,
                     V_SDI_best: float, mu_SDI_best: float, SDI_best: np.ndarray,
                     n_iterations: int, n_evals: int, elapsed_time: float):

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Otimização CMA-ES (ROBUSTA) para \\tilde{J}(P) ===\n\n")
        f.write(f"Arquivo de entrada: {csv_path}\n")
        f.write(f"Número de pontos da grade: {rsrp_mat.shape[0]}\n")
        f.write(f"Número de células (K): {K}\n")
        f.write(f"Seed: {seed}\n\n")

        f.write("Parâmetros do modelo:\n")
        f.write(f"  w_cov        = {w_cov}\n")
        f.write(f"  w_dom        = {w_dom}\n")
        f.write(f"  w_SDI        = {w_SDI}\n")
        f.write(f"  lambda       = {lambda_reg}\n")
        f.write(f"  S_min        = {S_min} dBm\n")
        f.write(f"  S_max        = {S_max} dBm\n")
        f.write(f"  Delta_ref    = {Delta_ref} dB\n")
        f.write(f"  sigma_P      = {sigma_P} dB\n")
        f.write(f"  RSRP_min_SDI = {RSRP_min_SDI} dBm\n")
        f.write(f"  sigma_SDI    = {sigma_SDI:.4f} m\n\n")

        f.write("Resultados do CMA-ES:\n")
        f.write(f"  Melhor J_til(P) encontrado: {J_til_best:.6f}\n")
        f.write(f"  J_base na solução ótima:   {J_base_best:.6f}\n")
        f.write(f"  R(P) na solução ótima:     {R_best:.6f}\n")
        f.write(f"  V_SDI na solução ótima:    {V_SDI_best:.6f}\n")
        f.write(f"  mu_SDI na solução ótima:   {mu_SDI_best:.4f} m\n")
        f.write(f"  Iterações:                 {n_iterations}\n")
        f.write(f"  Avaliações de função:      {n_evals}\n")
        f.write(f"  Tempo total (s):           {elapsed_time:.2f}\n\n")
        f.write("Vetor de potências ótimo (P* contínuo por PCI):\n")
        for pci_name, dP in zip(PCI_COLS, best_P):
            f.write(f"  {pci_name}: {dP:.3f} dB\n")

        f.write("\n")

        f.write("SDI na solução ótima (em metros):\n")
        for pci_name, sdi_val in zip(PCI_COLS, SDI_best):
            f.write(f"  {pci_name}: {sdi_val:.2f} m\n")
        f.write("\n")

        f.write("Melhor solução contínua (antes da quantização):\n")
        for pci_name, dP in zip(PCI_COLS, best_cont):
            f.write(f"  {pci_name}: {dP:.3f} dB\n")

def write_resumo_melhorJ(out_path: str, seed: int, sigma_SDI: float,
                         J_til_best: float, J_base_best: float,
                         stats_cov: dict,
                         deltaP_cont: np.ndarray,
                         azul_mediana: int,
                         vermelho_mediana: int,
                         branco_mediana: int,
                         gt5_mediana: int,
                         gt5_max: int,
                         n_validas: int,
                         aprovado: bool,
                         N_SEEDS: int,
                         Jtil_mediana: float,
                         Jtil_min: float,
                         Jtil_max: float,
                         Jbase_mediana: float):

    n_white = int(stats_cov["sem_cobertura"])
    n_red   = int(stats_cov["com_interferencia"])
    n_blue  = int(stats_cov["sem_interferencia"])
    n_total = int(stats_cov["total"])

    # quantidade de células com |ajuste| > 5 dB
    n_pci_gt5 = int(np.sum(np.abs(deltaP_cont) > 5.0))

    # percentuais
    p_white = 100.0 * n_white / n_total if n_total else 0.0
    p_red   = 100.0 * n_red   / n_total if n_total else 0.0
    p_blue  = 100.0 * n_blue  / n_total if n_total else 0.0
    n_min_validas = int(np.ceil(0.8 * N_SEEDS))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("-----------------\n")
        f.write(f"Seed: {seed}\n\n")

        f.write("Parâmetros do modelo:\n")
        f.write(f"w_cov        = {w_cov}\n")
        f.write(f"w_dom        = {w_dom}\n")
        f.write(f"w_SDI        = {w_SDI}\n")
        f.write(f"lambda       = {lambda_reg}\n")
        f.write(f"S_min        = {S_min} dBm\n")
        f.write(f"S_max        = {S_max} dBm\n")
        f.write(f"Delta_ref    = {Delta_ref} dB\n")
        f.write(f"sigma_P      = {sigma_P} dB\n")
        f.write(f"RSRP_min_SDI = {RSRP_min_SDI} dBm\n")
        f.write(f"sigma_SDI    = {sigma_SDI:.4f} m\n\n")

        f.write("Resultados do CMA-ES:\n")
        f.write(f"Melhor J_til(P) encontrado: {J_til_best:.6f}\n")
        f.write(f"J_base na solução ótima:   {J_base_best:.6f}\n\n")

        f.write("Cobertura:\n")
        f.write(f"Sem cobertura (branco): {n_white} ({p_white:.2f}%)\n")
        f.write(f"Cobertura com baixa dominância (vermelho): {n_red} ({p_red:.2f}%)\n")
        f.write(f"Cobertura com alta dominância (azul): {n_blue} ({p_blue:.2f}%)\n")
        f.write(f"Quantidade de PCI > +/- 5 dB: {n_pci_gt5}\n")
        # ----- PEDAÇO NOVO DE CÓDIGO ------
        f.write("\nAvaliação robusta (multi-seed):\n")
        f.write(f"Azul (mediana):     {azul_mediana}\n")
        f.write(f"Vermelho (mediana): {vermelho_mediana}\n")
        f.write(f"Branco (mediana):   {branco_mediana}\n")
        f.write(f"PCI |ΔP|>5 dB:      mediana={gt5_mediana} | max={gt5_max}\n")
        f.write(f"Seeds válidas:      {n_validas}/{N_SEEDS}\n")
        f.write(f"Critério de validade: ≥80% das seeds válidas\n")
        f.write(f"Aprovado (≥{n_min_validas}/{N_SEEDS}):    {aprovado}\n")
        f.write("\nMétricas do funcional (robustas):\n")
        f.write(f"J_til (mediana):    {Jtil_mediana:.6f}\n")
        f.write(f"J_til (min, max):   [{Jtil_min:.6f}, {Jtil_max:.6f}]\n")
        f.write(f"J_base (mediana):   {Jbase_mediana:.6f}\n")

        # ----- FIM DO PEDAÇO NOVO DE CÓDIGO ------
        f.write("-----------------\n")


def main():
    # Preparação
    lat, lon, rsrp_mat = load_data(CSV_PATH)
    x_m, y_m = latlon_to_xy(lat, lon)

    delta0 = np.zeros(K)

    RSRP_orig = compute_RSRP_max(rsrp_mat, delta0)

    export_coverage_csv(
        lat, lon, RSRP_orig,
        os.path.join(LOG_DIR, "Cobertura_Original.csv")
    )

    plot_rsrp_map(
        lat, lon, RSRP_orig,
        os.path.join(LOG_DIR, "Mapa_Cobertura_Original.png"),
        "Mapa de Cobertura Original (RSRP máximo)"
    )

    deltaP_zero = np.zeros(K, dtype=float)
    SDI0, mu_SDI0, _ = compute_SDI_for_P(rsrp_mat, x_m, y_m, deltaP_zero, rsrp_min=RSRP_min_SDI)
    sigma_SDI = float(mu_SDI0)

    def objective(deltaP_cont):
        J_til, *_ = compute_Jtil(rsrp_mat, x_m, y_m, deltaP_cont, sigma_SDI)
        return -J_til

    idx = find_next_idx(LOG_DIR)
    robust_log_path = os.path.join(LOG_DIR, f"CMAES_robusto_log_{idx:03d}.txt")
    summary_path     = os.path.join(LOG_DIR, f"CMAES_robusto_summary_{idx:03d}.csv")
    deltas_path      = os.path.join(LOG_DIR, f"CMAES_robusto_deltas_{idx:03d}.csv")
    bestJ_log_path = os.path.join(LOG_DIR, f"CMAES_melhorJ_log_{idx:03d}.txt")

    # (1) Plot ANTES da otimização (ΔP = 0)
    delta0 = np.zeros(K, dtype=float)
    out1 = os.path.join(LOG_DIR, f"Mapa_Cobertura_Inicial_{idx:03d}.png")
    stats1 = plot_coverage_map(
        lat=lat, lon=lon, rsrp_mat=rsrp_mat,
        deltaP_cont=delta0,
        out_path=out1,
        title="Mapa de Cobertura - Inicial (ΔP = 0)"
    )
    print("[MAPA] Inicial:", out1, stats1)

    # Executar múltiplas seeds
    x0 = np.zeros(K, dtype=float)
    runs = []

    for s in range(SEED_START, SEED_START + N_SEEDS):
        opts = {
            "bounds": [BOUNDS[0], BOUNDS[1]],
            "maxiter": MAXITER,
            "seed": int(s),
            "verb_log": 0,
            "verb_disp": 0,
        }

        start_time = time.time()
        es = cma.CMAEvolutionStrategy(x0, SIGMA0, opts)
        es.optimize(objective)
        print("STOP:", es.stop())  # <- mostra quais critérios dispararam
        elapsed = time.time() - start_time
        res = es.result
        best_cont = np.array(res.xbest, dtype=float)

        J_til, best_P, J_base, R_P, SDI, mu_SDI, V_SDI = compute_Jtil(
            rsrp_mat, x_m, y_m, best_cont, sigma_SDI
        )
        normP = float(np.linalg.norm(best_P, ord=2))

        stats_seed = plot_coverage_map(
            lat=lat, lon=lon, rsrp_mat=rsrp_mat,
            deltaP_cont=best_cont,
            out_path=os.path.join(LOG_DIR, f"_tmp_seed_{s}.png"),
            title=f"tmp seed {s}"
        )
        qtd_gt5_seed = int(np.sum(np.abs(best_cont) > 5.0))
        valida_seed = (qtd_gt5_seed <= 6)

        runs.append({
            "seed": s,
            "J_til": float(J_til),
            "J_base": float(J_base),
            "R_P": float(R_P),
            "V_SDI": float(V_SDI),
            "mu_SDI": float(mu_SDI),
            "iterations": int(res.iterations),
            "evaluations": int(res.evaluations),
            "elapsed_s": float(elapsed),
            "normP": normP,
            "best_cont": best_cont,
            "best_P": best_P,
            "SDI": SDI,
            "azul": stats_seed["sem_interferencia"],
            "vermelho": stats_seed["com_interferencia"],
            "branco": stats_seed["sem_cobertura"],
            "qtd_gt5": qtd_gt5_seed,
            "valida": valida_seed

        })

        if SAVE_PER_SEED_LOGS:
            per_seed_path = os.path.join(LOG_DIR, f"CMAES_robusto_log_{idx:03d}_seed_{s:03d}.txt")
            write_single_log(
                path=per_seed_path, seed=s, csv_path=CSV_PATH, rsrp_mat=rsrp_mat,
                sigma_SDI=sigma_SDI, best_cont=best_cont, best_P=best_P,
                J_til_best=J_til, J_base_best=J_base, R_best=R_P,
                V_SDI_best=V_SDI, mu_SDI_best=mu_SDI, SDI_best=SDI,
                n_iterations=int(res.iterations), n_evals=int(res.evaluations),
                elapsed_time=float(elapsed))

    # --------------------------------------------------
    # Agregação robusta (mediana nas seeds) - APÓS o loop
    # --------------------------------------------------
    azuis = np.array([r["azul"] for r in runs], dtype=float)
    vermelhos = np.array([r["vermelho"] for r in runs], dtype=float)
    brancos = np.array([r["branco"] for r in runs], dtype=float)
    gt5 = np.array([r["qtd_gt5"] for r in runs], dtype=float)
    validas = np.array([r["valida"] for r in runs], dtype=bool)

    azul_mediana = int(np.median(azuis))
    vermelho_mediana = int(np.median(vermelhos))
    branco_mediana = int(np.median(brancos))
    gt5_mediana = int(np.median(gt5))
    gt5_max = int(np.max(gt5))
    n_validas = int(np.sum(validas))
    n_min_validas = int(np.ceil(0.8 * N_SEEDS))
    aprovado = (n_validas >= n_min_validas)
    Jtils = np.array([r["J_til"] for r in runs], dtype=float)
    Jbases = np.array([r["J_base"] for r in runs], dtype=float)

    Jtil_mediana = float(np.median(Jtils))
    Jtil_min = float(np.min(Jtils))
    Jtil_max = float(np.max(Jtils))

    Jbase_mediana = float(np.median(Jbases))

    print("\n===== AVALIAÇÃO ROBUSTA (multi-seed) =====")
    print(f"Azul (mediana):     {azul_mediana}")
    print(f"Vermelho (mediana): {vermelho_mediana}")
    print(f"Branco (mediana):   {branco_mediana}")
    print(f"PCI |ΔP|>5 dB:      mediana={gt5_mediana} | max={gt5_max}")
    print(f"Seeds válidas:      {n_validas}/{N_SEEDS}")
    print(f"APROVADO (≥{n_min_validas}/{N_SEEDS}):    {aprovado}")
    print(f"J_til (mediana):    {Jtil_mediana:.6f}  (min={Jtil_min:.6f}, max={Jtil_max:.6f})")
    print(f"J_base (mediana):   {Jbase_mediana:.6f}")
    print("=========================================\n")

    # Summary por seed (sem arrays grandes)
    df = pd.DataFrame([{
        "seed": r["seed"],
        "J_til": r["J_til"],
        "J_base": r["J_base"],
        "R_P": r["R_P"],
        "V_SDI": r["V_SDI"],
        "mu_SDI": r["mu_SDI"],
        "iterations": r["iterations"],
        "evaluations": r["evaluations"],
        "elapsed_s": r["elapsed_s"],
        "normP": r["normP"],
    } for r in runs])

    # --- Melhor J_til absoluto (independente de erros) ---
    idx_best_J = int(df["J_til"].idxmax())
    bestJ_seed = int(df.loc[idx_best_J, "seed"])
    bestJ_run = next(r for r in runs if r["seed"] == bestJ_seed)

    # (2) Plot DEPOIS da otimização: corrigido com delta_P contínuo do melhor J̃
    delta_bestJ_cont = bestJ_run["best_cont"]

    out2 = os.path.join(LOG_DIR, f"Mapa_Cobertura_Corrigida_MelhorJ_{idx:03d}.png")
    stats2 = plot_coverage_map(
        lat=lat, lon=lon, rsrp_mat=rsrp_mat,
        deltaP_cont=delta_bestJ_cont,
        out_path=out2,
        title=f"Mapa de Cobertura - Corrigida (melhor J̃, seed={bestJ_seed})"
    )
    print("[MAPA] Corrigido (melhor J̃):", out2, stats2)

    # --- Resumo do melhor J_til em TXT com nome parametrizado ---
    resumo_fname = f"Resumo_wcov_{w_cov}_wdom_{w_dom}_wsdi_{w_SDI}_lambda_{lambda_reg}.txt"
    resumo_path = os.path.join(LOG_DIR, resumo_fname)

    write_resumo_melhorJ(
        out_path=resumo_path,
        seed=bestJ_seed,
        sigma_SDI=sigma_SDI,
        J_til_best=bestJ_run["J_til"],
        J_base_best=bestJ_run["J_base"],
        stats_cov=stats2,
        deltaP_cont=delta_bestJ_cont,
        azul_mediana=azul_mediana,
        vermelho_mediana=vermelho_mediana,
        branco_mediana=branco_mediana,
        gt5_mediana=gt5_mediana,
        gt5_max=gt5_max,
        n_validas=n_validas,
        aprovado=aprovado,
        N_SEEDS=N_SEEDS,
        Jtil_mediana=Jtil_mediana,
        Jtil_min=Jtil_min,
        Jtil_max=Jtil_max,
        Jbase_mediana=Jbase_mediana

    )
    print("[RESUMO] Gerado:", resumo_path)

    # --- (NOVO) Resumo no console ---
    qtd_azuis = int(stats2["sem_interferencia"])
    qtd_vermelhos = int(stats2["com_interferencia"])
    jtil = float(bestJ_run["J_til"])
    qtd_gt5 = int(np.sum(np.abs(delta_bestJ_cont) > 5.0))

    print("\n===== RESUMO (melhor J̃) =====")
    print(f"Quantidade de azuis: {qtd_azuis}")
    print(f"Quantidade de vermelhos: {qtd_vermelhos}")
    print(f"J_til(P): {jtil:.6f}")
    print(f"Quantidade de PCI > +/- 5 dB: {qtd_gt5}")
    print("=============================\n")

    delta_best = bestJ_run["best_cont"]

    RSRP_corr = compute_RSRP_max(rsrp_mat, delta_best)

    export_coverage_csv(
        lat, lon, RSRP_corr,
        os.path.join(LOG_DIR, "Cobertura_Corrigida.csv")
    )

    plot_rsrp_map(
        lat, lon, RSRP_corr,
        os.path.join(LOG_DIR, "Mapa_Cobertura_Corrigida.png"),
        "Mapa de Cobertura Corrigida (RSRP máximo)"
    )

    plot_rsrp_difference_map(
        lat, lon,
        RSRP_orig, RSRP_corr,
        os.path.join(LOG_DIR, "Mapa_Diferenca_Cobertura.png"),
        "Diferença de Cobertura (Corrigida − Original)"
    )
    df_sorted = df.sort_values(by=["J_til", "normP"], ascending=[False, True]).reset_index(drop=True)

    df_sorted.to_csv(summary_path, index=False)

    best_seed = int(df_sorted.loc[0, "seed"])
    best_run  = next(r for r in runs if r["seed"] == best_seed)

    # Robustez estatística
    J_vals = df["J_til"].to_numpy(dtype=float)
    J_mean = float(np.mean(J_vals))
    J_std  = float(np.std(J_vals, ddof=1)) if len(J_vals) > 1 else 0.0
    ci_lo, ci_hi = bootstrap_ci_mean(J_vals)
    all_cont = np.vstack([r["best_cont"] for r in runs])
    # Dispersão de deltas por PCI
    df_deltas = pd.DataFrame({
        "PCI": PCI_COLS,
        "deltaP_mean": np.mean(all_cont, axis=0),
        "deltaP_std":  np.std(all_cont, axis=0, ddof=1) if N_SEEDS > 1 else np.zeros(K),
    }).sort_values("deltaP_std", ascending=False)
    df_deltas.to_csv(deltas_path, index=False)

    # Log robusto final (melhor seed + resumo robustez)
    write_single_log(
        path=robust_log_path, seed=best_seed, csv_path=CSV_PATH, rsrp_mat=rsrp_mat,
        sigma_SDI=sigma_SDI,
        best_cont=best_run["best_cont"],
        best_P=best_run["best_P"],
        J_til_best=best_run["J_til"], J_base_best=best_run["J_base"], R_best=best_run["R_P"],
        V_SDI_best=best_run["V_SDI"], mu_SDI_best=best_run["mu_SDI"], SDI_best=best_run["SDI"],
        n_iterations=best_run["iterations"], n_evals=best_run["evaluations"],
        elapsed_time=best_run["elapsed_s"])
    write_single_log(
        path=bestJ_log_path,
        seed=bestJ_seed,
        csv_path=CSV_PATH,
        rsrp_mat=rsrp_mat,
        sigma_SDI=sigma_SDI,
        best_cont=bestJ_run["best_cont"],
        best_P=bestJ_run["best_P"],
        J_til_best=bestJ_run["J_til"],
        J_base_best=bestJ_run["J_base"],
        R_best=bestJ_run["R_P"],
        V_SDI_best=bestJ_run["V_SDI"],
        mu_SDI_best=bestJ_run["mu_SDI"],
        SDI_best=bestJ_run["SDI"],
        n_iterations=bestJ_run["iterations"],
        n_evals=bestJ_run["evaluations"],
        elapsed_time=bestJ_run["elapsed_s"],
    )

    with open(bestJ_log_path, "a", encoding="utf-8") as f:
        f.write("\n\n=== Nota ===\n")
        f.write("Este arquivo registra a solução com maior J_til(P) entre todas as seeds,\n")
        f.write("Use-o para comparação com a solução robusta selecionada.\n")

    # Acrescentar seção de robustez ao log robusto
    with open(robust_log_path, "a", encoding="utf-8") as f:
        f.write("\n\n=== Robustez (multi-seed) ===\n")
        f.write(f"Total de seeds: {N_SEEDS} (seed_start={SEED_START})\n")
        f.write("Métricas globais:\n")
        f.write(f"  J_til média: {J_mean:.6f}\n")
        f.write(f"  J_til desvio-padrão: {J_std:.6f}\n")
        f.write(f"  IC95% bootstrap da média J_til: [{ci_lo:.6f}, {ci_hi:.6f}]\n\n")
        f.write("Arquivos gerados:\n")
        f.write(f"  Summary por seed:   {summary_path}\n")
        f.write(f"  Deltas mean/std:    {deltas_path}\n")
        if SAVE_PER_SEED_LOGS:
            f.write("  Logs por seed:      CMAES_robusto_log_{IDX:03d}_seed_{SEED:03d}.txt\n")


    print("Otimização robusta concluída.")
    print(f"Log robusto final: {robust_log_path}")
    print(f"Summary por seed:  {summary_path}")
    print(f"Deltas mean/std:   {deltas_path}")
    print(f"Melhor seed (por J_til): {best_seed}")
    print(f"Melhor seed por J_til: {bestJ_seed} -> log: {bestJ_log_path}")

if __name__ == "__main__":
    main()
