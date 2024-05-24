# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:09:38 2024

@author: Jaime
"""


# =============================================================================
# EJEMPLO 2 DE MINERÍA DE DATOS CON EL DATASET WINE APLICANDO ANÁLISIS DE REGRESIÓN
#==============================================================================

# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA

# Ajustes generales de estilo para las gráficas
sns.set(style="whitegrid")

# Paso 1: Cargar el dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
df = pd.read_csv(url, sep=';')

# Asumimos que la variable 'type' está presente en el dataset. Si no, debemos agregarla de alguna manera.
# Aquí solo se trabaja con un dataset, así que se puede crear una variable 'type' para demostración
df['type'] = 'white'  # Ya que este dataset es de vino blanco

# Mostrar las primeras filas del dataset para verificar que se ha cargado correctamente
print("Primeras filas del dataset:")
print(df.head())

# Mostrar la información general del dataset para entender la estructura de los datos
print("\nInformación del dataset:")
print(df.info())

# Paso 2: Análisis Estadístico Descriptivo Básico
# Medidas de tendencia central: media, mediana y moda
print("\nMedidas de tendencia central:")
mean_values = df.mean()
median_values = df.median()
mode_values = df.mode().iloc[0]

print("Media:\n", mean_values)
print("Mediana:\n", median_values)
print("Moda:\n", mode_values)

# Medidas de dispersión: varianza, desviación estándar y rango
print("\nMedidas de dispersión:")
numerical_df = df.select_dtypes(include=[np.number])
variance_values = numerical_df.var()
std_dev_values = numerical_df.std()
range_values = numerical_df.max() - numerical_df.min()

print("Varianza:\n", variance_values)
print("Desviación Estándar:\n", std_dev_values)
print("Rango:\n", range_values)

# Cuartiles
print("\nCuartiles:")
quartiles = numerical_df.quantile([0.25, 0.5, 0.75])
print("Cuartiles:\n", quartiles)

# Sesgo (Skewness)
print("\nSesgo (Skewness):")
skewness_values = numerical_df.apply(lambda x: skew(x.dropna()))
print(skewness_values)

# Curtosis (Kurtosis)
print("\nCurtosis (Kurtosis):")
kurtosis_values = numerical_df.apply(lambda x: kurtosis(x.dropna()))
print(kurtosis_values)

# Paso 3: Visualización de Datos

# Diagramas de Dispersión (scatter plots)
plt.figure(figsize=(12, 10))
sns.pairplot(df, diag_kind='kde', markers='+')
plt.suptitle('Diagramas de Dispersión entre las Variables Numéricas', y=1.02)
plt.show()

# Histogramas
plt.figure(figsize=(14, 10))
df[numerical_df.columns].hist(bins=20, figsize=(14, 10), layout=(4, 3), edgecolor='black')
plt.suptitle('Histogramas de las Variables Numéricas', y=1.02)
plt.tight_layout()
plt.show()

# Diagramas de Caja (box plots) de cada variable numérica
numerical_vars = df.select_dtypes(include=[np.number]).columns

for var in numerical_vars:
    plt.figure(figsize=(14, 8))
    sns.boxplot(y=df[var])
    plt.title(f'Diagrama de Caja de {var}')
    plt.show()

# Diagramas de Caja por Variable Categórica (quality)
for var in numerical_vars:
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='quality', y=var, data=df)
    plt.title(f'Diagrama de Caja de {var} por Calidad del Vino')
    plt.show()

# Paso 4: Construir Tablas de Distribución de Frecuencias

# Definir función para calcular y mostrar la tabla de distribución de frecuencias
def frequency_table(df, variable, bins=10):
    frequency, bin_edges = np.histogram(df[variable], bins=bins)
    bin_edges = np.round(bin_edges, 2)
    freq_table = pd.DataFrame({'Bin': [f"{bin_edges[i]} - {bin_edges[i+1]}" for i in range(len(bin_edges)-1)],
                               'Frequency': frequency})
    return freq_table

# Generar y mostrar tablas de distribución de frecuencias para cada variable numérica
numerical_variables = df.select_dtypes(include=[np.number]).columns
for variable in numerical_variables:
    print(f"\nTabla de distribución de frecuencias para {variable}:")
    freq_table = frequency_table(df, variable)
    print(freq_table)

# Paso 5: Pre-procesar los Datos

# Separar las variables independientes (X) y la variable dependiente (y)
X = df.drop(columns=['quality'])
y = df['quality']

# Separar las variables numéricas y categóricas
X_numeric = X.select_dtypes(include=[np.number])
X_categorical = X.select_dtypes(exclude=[np.number])

# Estandarizar las características numéricas
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# Combinar las características numéricas estandarizadas y las categóricas
X_processed = np.concatenate([X_numeric_scaled, pd.get_dummies(X_categorical).values], axis=1)

# Dividir el dataset en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

# Añadir una constante al conjunto de entrenamiento y prueba (necesario para statsmodels)
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Paso 6: Entrenar el Modelo de Regresión Lineal Múltiple

# Entrenar el modelo de regresión lineal múltiple con statsmodels
model = sm.OLS(y_train, X_train).fit()

# Mostrar el resumen del modelo
print(model.summary())

# Predecir los valores en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo con métricas adicionales
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")

# Visualizar la comparación de valores reales vs predichos
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Comparación de Valores Reales y Predichos')
plt.show()

# Distribución de los residuos
plt.figure(figsize=(10, 6))
sns.histplot(y_test - y_pred, kde=True, bins=30)
plt.title('Distribución de los Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frecuencia')
plt.show()

# Paso 7: Análisis de Componentes Principales (PCA) y Diagrama de Dispersión
# Aplicar PCA para reducir la dimensionalidad
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_numeric_scaled)

# Crear un DataFrame con las dos primeras componentes principales
df_pca = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])

# Agregar la variable 'quality' al DataFrame
df_pca['quality'] = y.values

# Diagrama de dispersión de las dos primeras componentes principales
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='quality', palette='viridis', data=df_pca, alpha=0.6)
plt.title('Diagrama de Dispersión de las Dos Primeras Componentes Principales')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Quality')
plt.show()
