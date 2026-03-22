"""
Hecho por : Rocha Cantu Nidia Wendoly  Fecha: 22  de Marzo 2026
Clase: Inteligencia artificial y su ética - Tema 4.3 Aprendizaje automático - Actividad 21
MIA - Intituto Tecnológico de Nuevo Laredo - Prof. Carlos Arturo Guerrero Crespo
Titulo: Sistema de Recomendación de Películas
Descripción:
Agrupa usuarios mediante clustering y recomienda películas
basadas en preferencias similares.
"""

# =========================
# LIBRERÍAS
# =========================
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

# =========================
# 1. GENERAR DATOS
# =========================
def generar_datos():
    """
    Simula calificaciones de usuarios a películas.
    """
    np.random.seed(42)

    usuarios = 200
    peliculas = 50

    data = np.random.randint(1, 6, size=(usuarios, peliculas))

    print("Datos generados:")
    print(f"Usuarios: {usuarios}, Películas: {peliculas}")

    return pd.DataFrame(data)


# =========================
# 2. PREPROCESAMIENTO
# =========================
def preprocesar_datos(df):
    """
    Escala y reduce dimensionalidad.
    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=10)
    df_pca = pca.fit_transform(df_scaled)

    return df_pca


# =========================
# 3. MODELADO
# =========================
def aplicar_modelos(X):
    """
    Aplica múltiples algoritmos de clustering.
    """
    modelos = {
        "KMeans": KMeans(n_clusters=5, random_state=42),
        "Agglomerative": AgglomerativeClustering(n_clusters=5),
        "DBSCAN": DBSCAN(eps=3, min_samples=5)
    }

    resultados = {}

    for nombre, modelo in modelos.items():
        etiquetas = modelo.fit_predict(X)

        # Evitar error en DBSCAN
        if len(set(etiquetas)) > 1:
            score = silhouette_score(X, etiquetas)
        else:
            score = -1

        print(f"{nombre} - Silhouette Score: {score:.4f}")
        resultados[nombre] = (modelo, etiquetas, score)

    return resultados


# =========================
# 4. RECOMENDACIÓN
# =========================
def recomendar_peliculas(df, etiquetas):
    """
    Genera recomendaciones por cluster.
    """
    df['cluster'] = etiquetas

    recomendaciones = {}

    for cluster in np.unique(etiquetas):
        grupo = df[df['cluster'] == cluster]

        promedio = grupo.drop('cluster', axis=1).mean()
        top_peliculas = promedio.sort_values(ascending=False).head(5)

        recomendaciones[cluster] = top_peliculas.index.tolist()

    return recomendaciones


# =========================
# 5. MAIN
# =========================
def main():
    # Datos
    df = generar_datos()

    # Preprocesamiento
    X = preprocesar_datos(df)

    # Modelos
    resultados = aplicar_modelos(X)

    # Seleccionar mejor modelo (KMeans)
    mejor = resultados["KMeans"]
    etiquetas = mejor[1]

    # Recomendaciones
    recomendaciones = recomendar_peliculas(df, etiquetas)

    print("\nRecomendaciones por cluster:")
    for cluster, pelis in recomendaciones.items():
        print(f"Cluster {cluster}: {pelis}")


if __name__ == "__main__":
    main()