import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset y omitir líneas problemáticas
data = pd.read_csv("dataPython.csv", sep=';', on_bad_lines='skip')

# Ver las primeras filas del dataset para entender qué representan las columnas
print("Primeras filas del dataset:")
print(data.head())

# Verificar los nombres de las columnas
print("\nColumnas originales:")
print(data.columns)

# Ajusta los nombres de las columnas para que coincidan con el número de columnas en tu dataset
data.columns = ['fecha_factura', 'documento', 'grupo_cliente', 'provincia',
                'grupo_producto', 'marca', 'cantidad', 'valor_unitario', 
                'valor_total', 'descuento', 'iva', 'clasificador', 'otra_columna']

# Verificar que las columnas ahora se corresponden con lo que necesitas
print("\nColumnas después de renombrar:")
print(data.columns)

# Limpiar la columna 'valor_total' reemplazando los puntos de miles y comas
def clean_value(value):
    # Reemplazar puntos de miles por nada y las comas por puntos decimales
    value = str(value).replace('.', '')  # Eliminar puntos de miles
    value = value.replace(',', '.')      # Asegurarse de que las comas sean puntos decimales
    return value

# Aplicar la limpieza y convertir a float
data['valor_total'] = data['valor_total'].apply(clean_value).astype(float)

# Codificar variables categóricas para análisis de correlación
categorical_features = ['grupo_cliente', 'provincia', 'marca', 'grupo_producto']

# Verificar si las columnas categóricas existen
for col in categorical_features:
    if col not in data.columns:
        print(f"Columna '{col}' no encontrada en los datos.")
    else:
        data[col] = data[col].astype("category").cat.codes

# Selección de columnas relevantes para análisis
features = ['cantidad', 'valor_unitario', 'valor_total', 'descuento', 
            'iva', 'grupo_cliente', 'provincia', 'marca', 'grupo_producto']

# Comprobar que todas las columnas necesarias están presentes
missing_columns = [col for col in features if col not in data.columns]
if missing_columns:
    print(f"Las siguientes columnas están faltando: {missing_columns}")
else:
    # Muestreo aleatorio del 10% de los datos para prueba
    sample_data = data.sample(frac=0.1, random_state=42)
    
    # Matriz de correlación entre features
    corr_features = sample_data[features].corr(method='pearson')

    # Visualizar matriz de correlación entre features
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_features, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlación entre Features")
    plt.show()

    # Función para graficar distribuciones de probabilidad basadas en features
    def plot_distributions(df, features):
        rows = (len(features) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
        axes = axes.flatten()

        for idx, col in enumerate(features):
            # Usar kde=False para evitar la estimación KDE si los datos son muy grandes
            sns.histplot(data=df, x=col, kde=False, ax=axes[idx], color='blue')
            axes[idx].set_title(f'Distribución de {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frecuencia')

        # Eliminar gráficos vacíos si el número de columnas no es múltiplo de 3
        for idx in range(len(features), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()

    # Llamar a la función para graficar las distribuciones
    plot_distributions(sample_data, features)
