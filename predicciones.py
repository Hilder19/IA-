import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model

# Función para obtener datos de mercado en tiempo real usando Alpha Vantage API
def obtener_datos_tiempo_real(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={api_key}"
    response = requests.get(url)
    data = response.json()

    # Extraer los datos
    time_series = data['Time Series (1min)']
    df = pd.DataFrame(time_series).transpose()
    df.columns = ['Apertura', 'Máximo', 'Mínimo', 'Cierre', 'Volumen']
    df.index = pd.to_datetime(df.index)

    # Convertir a valores numéricos
    df = df.astype(float)
    
    return df

# Cargar modelo
def cargar_modelo(path):
    return load_model(path)

# Función para realizar la predicción
def predecir_precio(modelo, datos_entrada):
    # Escalar los datos de entrada
    scaler = MinMaxScaler(feature_range=(0, 1))
    datos_entrada_scaled = scaler.fit_transform(datos_entrada)

    # Redimensionar los datos para que sean compatibles con el modelo
    datos_entrada_scaled = np.reshape(datos_entrada_scaled, (1, datos_entrada_scaled.shape[0], datos_entrada_scaled.shape[1]))

    # Realizar la predicción
    prediccion = modelo.predict(datos_entrada_scaled)
    
    # Invertir la escala para obtener el precio en valor real
    prediccion_invertida = scaler.inverse_transform(prediccion)
    
    return prediccion_invertida

from modelo_creacion.crear_modelo import cargar_modelo  # Asegúrate de tener esta función
import numpy as np

def predecir_precio(modelo, datos_entrada):
    datos_entrada = np.reshape(datos_entrada, (1, datos_entrada.shape[0], datos_entrada.shape[1]))
    return modelo.predict(datos_entrada)
