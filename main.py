import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def cargar_datos(ruta_csv):
    try:
        data = pd.read_csv(ruta_csv)
        print("Datos cargados correctamente.")
        print(data.head())  
        return data
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{ruta_csv}'. Asegúrate de que esté en la misma carpeta.")
        return None

def mostrar_matriz_correlacion(data):
    print("Calculando la matriz de correlación...")

    data_numerica = data.select_dtypes(include=['float64', 'int64'])
    corr_matrix = data_numerica.corr()  
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Matriz de Correlación")
    plt.show()

def separar_clases(data, columna_objetivo):
    print(f"Separando datos por la columna objetivo: '{columna_objetivo}'...")
    if columna_objetivo not in data.columns:
        print(f"Error: La columna '{columna_objetivo}' no existe en el dataset.")
        return None, None
    class_0 = data[data[columna_objetivo] == 0]
    class_1 = data[data[columna_objetivo] == 1]
    print(f"Clase 0: {len(class_0)} filas, Clase 1: {len(class_1)} filas.")
    return class_0, class_1

def graficar_distribucion(data):
    print("Graficando la distribución de las variables...")
    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']: 
            plt.hist(data[column], bins=30, alpha=0.5, label=column)
            plt.title(f"Distribución de {column}")
            plt.legend()
            plt.show()

if __name__ == "__main__":
    archivo_csv = 'holaAmigos.csv'

    dataset = cargar_datos(archivo_csv)

    if dataset is not None:
        mostrar_matriz_correlacion(dataset)

        columna_objetivo = 'target' 
        class_0, class_1 = separar_clases(dataset, columna_objetivo)

        graficar_distribucion(dataset)
