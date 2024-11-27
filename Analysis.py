import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

data = pd.read_csv("dataPython.csv", sep=';', on_bad_lines='skip')

print("Primeras filas del dataset:")
print(data.head())

print("\nColumnas originales:")
print(data.columns)

data.columns = ['fecha_factura', 'documento', 'grupo_cliente', 'provincia',
                'grupo_producto', 'marca', 'cantidad', 'valor_unitario', 
                'valor_total', 'descuento', 'iva', 'clasificador', 'otra_columna']

print("\nColumnas después de renombrar:")
print(data.columns)

def clean_value(value):
    value = str(value).replace('.', '')  
    value = value.replace(',', '.')
    return value

data['valor_total'] = data['valor_total'].apply(clean_value).astype(float)

categorical_features = ['grupo_cliente', 'provincia', 'marca', 'grupo_producto']

for col in categorical_features:
    if col not in data.columns:
        print(f"Columna '{col}' no encontrada en los datos.")
    else:
        data[col] = data[col].astype("category").cat.codes

features = ['cantidad', 'valor_unitario', 'valor_total', 'descuento', 
            'iva', 'grupo_cliente', 'provincia', 'marca', 'grupo_producto']

missing_columns = [col for col in features if col not in data.columns]
if missing_columns:
    print(f"Las siguientes columnas están faltando: {missing_columns}")
else:
    sample_data = data.sample(frac=0.1, random_state=42)
    
    corr_features = sample_data[features].corr(method='pearson')

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_features, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlación entre Features")
    plt.show()

    def plot_distributions(df, features):
        rows = (len(features) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
        axes = axes.flatten()
        colors = sns.color_palette("husl", len(features))  

        for idx, col in enumerate(features):
            data_col = df[col]
            sns.histplot(data=data_col, kde=True, ax=axes[idx], color=colors[idx])
            
            mean = data_col.mean()
            median = data_col.median()

            axes[idx].set_title(
                f'Distribución de {col}\nMedia: {mean:.2f}, Mediana: {median:.2f}',
                fontsize=12
            )
            axes[idx].axvline(mean, color='red', linestyle='--', label='Media')
            axes[idx].axvline(median, color='green', linestyle='-', label='Mediana')
            axes[idx].legend()

            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frecuencia')


        for idx in range(len(features), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()

plot_distributions(sample_data, features)