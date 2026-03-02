import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score
import os

# Início do tempo de execução
horainicio = datetime.datetime.now()

# Solicita o PCI ao usuário
pci = input("PCI? (ex: 009) ")
best_alpha = float(input("Alpha? (ex: 0.0025) ")) #best_alpha = 0.0036

# Caminho para o arquivo de entrada
caminho_arquivo = f"/home/drcoliveira/UFU/Doutorado/Qualify/Projeto Definitivo/Dataset/Base Separada por PCI/PCI_{pci}.csv"

# Leitura dos dados
data = pd.read_csv(caminho_arquivo, sep=',')
X = data[['lat_norm', 'long_norm']]
y = data['RSRP_norm'].values

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

# Kernel e modelo de Processo Gaussiano
kernel = C(0.1, (1e-4, 1e1)) * RBF(length_scale=0.1, length_scale_bounds=(1e-4, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=best_alpha)

# Treinamento
gp.fit(X_train, y_train)

# Previsão no conjunto de teste
y_pred_normalized, sigma_normalized = gp.predict(X_test, return_std=True)
y_pred_normalized = np.clip(y_pred_normalized, 0, 1)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred_normalized)
r2 = r2_score(y_test, y_pred_normalized)
rmse = np.sqrt(mse)
horaatual = datetime.datetime.now()
tempodecorrido = horaatual - horainicio

print("\n_____________________ Resultados _____________________")
print("Tempo decorrido: ", tempodecorrido)
print(f'Erro Quadrático Médio no conjunto de teste: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R² no conjunto de teste: {r2}')

# Carrega a matriz de análise
matriz_analise_path = '/home/drcoliveira/UFU/Doutorado/Qualify/Projeto Definitivo/Dataset/Base Separada por PCI/matriz_analise_normalizado.csv'
matriz_analise = pd.read_csv(matriz_analise_path, sep=',')

# Predição
rsrp_predito_normalized, sigma_predito_normalized = gp.predict(matriz_analise[['lat_norm', 'long_norm']], return_std=True)
rsrp_predito_normalized = np.clip(rsrp_predito_normalized, 0, 1)

# Adiciona ao DataFrame
matriz_analise['RSRP_Predito_normalizado'] = rsrp_predito_normalized
matriz_analise['Sigma_Predito_normalizado'] = sigma_predito_normalized

# Salva o CSV com os resultados
saida_csv = f'/home/drcoliveira/UFU/Doutorado/Qualify/Projeto Definitivo/Dataset/Base Separada por PCI/predicao_rsrp_normalizado_{pci}.csv'
matriz_analise.to_csv(saida_csv, index=False, header=False)
print(f"Predições salvas em {saida_csv}")

# Gera a imagem com mapa de cobertura
plt.figure(figsize=(10, 8))
sc = plt.scatter(matriz_analise['long_norm'], matriz_analise['lat_norm'], c=rsrp_predito_normalized, cmap='plasma', s=10)
plt.colorbar(sc, label='RSRP Predito Normalizado')
plt.xlabel('Longitude Normalizada')
plt.ylabel('Latitude Normalizada')
plt.title(f'Mapa de Cobertura - PCI {pci}')

# Caminho para imagem
imagem_saida = f'/home/drcoliveira/UFU/Doutorado/Qualify/Projeto Definitivo/Dataset/Base Separada por PCI/mapa_cobertura_{pci}.png'
plt.savefig(imagem_saida, dpi=300)
plt.close()
print(f"Imagem de cobertura salva em {imagem_saida}")
