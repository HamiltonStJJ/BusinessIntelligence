import pandas as pd
import numpy as np

def crear_dataset_clasificado(filas, columnas, rango_min, rango_max):
    """
    Genera un dataset con valores aleatorios y una columna de clasificación (0 o 1).
    
    :param filas: Número de filas del dataset.
    :param columnas: Número de columnas numéricas.
    :param rango_min: Valor mínimo del rango de datos.
    :param rango_max: Valor máximo del rango de datos.
    :return: Un DataFrame generado.
    """
    print("Generando dataset clasificado...")

    datos_numericos = np.random.randint(rango_min, rango_max + 1, size=(filas, columnas))
    nombres_columnas = [f"Columna_{i+1}" for i in range(columnas)]

    df = pd.DataFrame(datos_numericos, columns=nombres_columnas)

    df['Clasificación'] = np.random.choice([0, 1], size=filas)

    print("Dataset generado con éxito:")
    print(df.head()) 
    return df


if __name__ == "__main__":
    print("Bienvenido al generador de dataset clasificado.")
    filas = int(input("¿Cuántas filas quieres? "))
    columnas = int(input("¿Cuántas columnas numéricas quieres? "))
    rango_min = int(input("Valor mínimo del rango de datos: "))
    rango_max = int(input("Valor máximo del rango de datos: "))

    dataset = crear_dataset_clasificado(filas, columnas, rango_min, rango_max)

    guardar_csv = input("¿Quieres guardar el dataset como un archivo CSV (Si/No)? ").strip().lower()
    if guardar_csv == 'si':
        nombre_archivo = input("Ingresa el nombre del archivo (sin extensión): ") + ".csv"
        dataset.to_csv(nombre_archivo, index=False)
        print(f"Dataset guardado exitosamente como '{nombre_archivo}'.")
