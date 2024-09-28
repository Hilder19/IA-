import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import psycopg2
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import matplotlib.pyplot as plt

# Conexión a la base de datos PostgreSQL
conn = psycopg2.connect(
    dbname="nombre_de_tu_base_de_datos",
    user="tu_usuario",
    password="tu_password",
    host="localhost"
)
cursor = conn.cursor()

# Función para obtener los últimos 40,320 registros
def obtener_datos_desde_bd():
    query = """
        SELECT 
            rate as Close, open as Open, high as High, low as Low
        FROM exchange_rates_history
        ORDER BY last_updated DESC
        LIMIT 40320;
    """
    cursor.execute(query)
    datos = cursor.fetchall()
    return pd.DataFrame(datos, columns=['Close', 'Open', 'High', 'Low'])

# Obtener datos para la red neuronal
data = obtener_datos_desde_bd()

# Subredes Neuronales
def calcular_macd(data, short_window=12, long_window=26, signal_window=9):
    data['EMA12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data['MACD'], data['Signal_Line']

def calcular_rsi(data, window=14):
    delta = data['Close'].diff()
    ganancia = delta.where(delta > 0, 0)
    perdida = -delta.where(delta < 0, 0)
    avg_ganancia = ganancia.rolling(window=window).mean()
    avg_perdida = perdida.rolling(window=window).mean()
    rs = avg_ganancia / avg_perdida
    return 100 - (100 / (1 + rs))

def es_engulfing(data):
    alcista = (data['Close'].shift(1) < data['Open'].shift(1)) & \
              (data['Open'] < data['Close'].shift(1)) & \
              (data['Close'] > data['Open'].shift(1))
    
    bajista = (data['Close'].shift(1) > data['Open'].shift(1)) & \
              (data['Open'] > data['Close'].shift(1)) & \
              (data['Close'] < data['Open'].shift(1))
    
    return alcista, bajista

def es_doji(data):
    doji = abs(data['Close'] - data['Open']) < (data['High'] - data['Low']) * 0.1
    return doji

def es_estrella_fugaz(data):
    estrella_fugaz = (data['High'] - data['Open']) > 2 * (data['Open'] - data['Close'])
    return estrella_fugaz

def es_martillo(data):
    martillo = (data['Open'] - data['Low']) > 2 * (data['Close'] - data['Open'])
    return martillo

def es_swing(data):
    swing = (data['High'].shift(1) > data['High']) & (data['Low'].shift(1) < data['Low'])
    return swing

def calcular_scalping(data):
    scalping = (data['Close'] - data['Open']) / data['Open'] * 100
    return scalping

def calcular_aroon(data, window=25):
    rolling_high = data['High'].rolling(window=window).apply(lambda x: np.argmax(x), raw=False)
    rolling_low = data['Low'].rolling(window=window).apply(lambda x: np.argmin(x), raw=False)
    aroon_up = (window - rolling_high) / window * 100
    aroon_down = (window - rolling_low) / window * 100
    return aroon_up, aroon_down

def calcular_parabolic_sar(data):
    data['PSAR'] = data['Close']  # Ejemplo temporal, implementar lógica real
    return data['PSAR']

def calcular_adx(data, window=14):
    data['TR'] = np.maximum(data['High'] - data['Low'], 
                            np.maximum(abs(data['High'] - data['Close'].shift(1)), 
                                       abs(data['Low'] - data['Close'].shift(1))))
    data['ATR'] = data['TR'].rolling(window=window).mean()
    data['+DM'] = np.where((data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low']), 
                           data['High'] - data['High'].shift(1), 0)
    data['-DM'] = np.where((data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1)), 
                           data['Low'].shift(1) - data['Low'], 0)
    data['+DI'] = 100 * (data['+DM'] / data['ATR'])
    data['-DI'] = 100 * (data['-DM'] / data['ATR'])
    data['DX'] = (abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI'])) * 100
    return data['DX'].rolling(window=window).mean()

def calcular_ao_ac(data):
    ao = data['Close'].rolling(window=5).mean() - data['Close'].rolling(window=34).mean()
    ac = ao.rolling(window=5).mean() - ao.rolling(window=34).mean()
    return ao, ac

def calcular_ichimoku(data):
    data['tenkan_sen'] = (data['High'].rolling(window=9).max() + data['Low'].rolling(window=9).min()) / 2
    data['kijun_sen'] = (data['High'].rolling(window=26).max() + data['Low'].rolling(window=26).min()) / 2
    data['senkou_span_a'] = (data['tenkan_sen'] + data['kijun_sen']) / 2
    data['senkou_span_b'] = (data['High'].rolling(window=52).max() + data['Low'].rolling(window=52).min()) / 2
    return data['tenkan_sen'], data['kijun_sen'], data['senkou_span_a'], data['senkou_span_b']

# Calcular indicadores técnicos
data['MACD'], data['Signal_Line'] = calcular_macd(data)
data['RSI'] = calcular_rsi(data)
data['Aroon_Up'], data['Aroon_Down'] = calcular_aroon(data)
data['Parabolic_SAR'] = calcular_parabolic_sar(data)
data['ADX'] = calcular_adx(data)
data['AO'], data['AC'] = calcular_ao_ac(data)
data['Tenkan_Sen'], data['Kijun_Sen'], data['Senkou_Span_A'], data['Senkou_Span_B'] = calcular_ichimoku(data)
data['Engulfing_Alcista'], data['Engulfing_Bajista'] = es_engulfing(data)
data['Doji'] = es_doji(data)
data['Estrella_Fugaz'] = es_estrella_fugaz(data)
data['Martillo'] = es_martillo(data)
data['Swing'] = es_swing(data)
data['Scalping'] = calcular_scalping(data)

# Filtrar las características importantes
features = data[['MACD', 'Signal_Line', 'RSI', 'Aroon_Up', 'Aroon_Down', 'Parabolic_SAR', 'ADX', 'AO', 'AC',
                 'Tenkan_Sen', 'Kijun_Sen', 'Senkou_Span_A', 'Senkou_Span_B', 'Engulfing_Alcista', 
                 'Engulfing_Bajista', 'Doji', 'Estrella_Fugaz', 'Martillo', 'Swing', 'Scalping']].fillna(0)
target = data['Close'].shift(-1).fillna(0)

# Escalar los datos
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# División en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, shuffle=False)

# Definir función de modelo para ajuste de hiperparámetros con Keras Tuner
def build_model(hp):
    model = models.Sequential()
    
    model.add(layers.Dense(units=hp.Int('units_layer_1', min_value=32, max_value=128, step=32),
                           activation='relu', input_shape=(X_train.shape[1],),
                           kernel_regularizer=regularizers.l2(0.001)))
    
    model.add(layers.Dense(units=hp.Int('units_layer_2', min_value=32, max_value=128, step=32),
                           activation='relu',
                           kernel_regularizer=regularizers.l2(0.001)))
    
    model.add(layers.Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Configuración de Keras Tuner para búsqueda de hiperparámetros
tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=5, executions_per_trial=3, 
                        directory='tuner_results', project_name='financial_forecast')

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Búsqueda de hiperparámetros
tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])

# Obtener los mejores hiperparámetros y entrenar el mejor modelo
best_hps = tuner.get_best_hyperparameters()[0]
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])

# Evaluar el modelo
test_loss, test_mae = best_model.evaluate(X_test, y_test)
print(f'MAE en datos de prueba: {test_mae}')

# Graficar la evolución del error durante el entrenamiento
plt.plot(history.history['loss'], label='Pérdida en entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida en validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Graficar MAE
plt.plot(history.history['mae'], label='MAE en Entrenamiento')
plt.plot(history.history['val_mae'], label='MAE en Validación')
plt.title('MAE en Entrenamiento y Validación')
plt.xlabel('Épocas')
plt.ylabel('MAE')
plt.legend()
plt.show()

# Graficar el historial de entrenamiento
def graficar_historial(history):
    plt.plot(history.history['loss'], label='Pérdida')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.plot(history.history['mae'], label='MAE')
    plt.plot(history.history['val_mae'], label='MAE de Validación')
    plt.title('Historial de Entrenamiento')
    plt.legend()
    plt.show()

graficar_historial(history)

# Cerrar la conexión a la base de datos
cursor.close()
conn.close()
