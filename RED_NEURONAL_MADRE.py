import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Subredes Neuronales

# 1. Red Neuronal MACD
def calcular_macd(data, short_window=12, long_window=26, signal_window=9):
    data['EMA12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data['MACD'], data['Signal_Line']

# 2. Red Neuronal Índice de Fuerza Relativa (RSI)
def calcular_rsi(data, window=14):
    delta = data['Close'].diff()
    ganancia = delta.where(delta > 0, 0)
    perdida = -delta.where(delta < 0, 0)
    avg_ganancia = ganancia.rolling(window=window).mean()
    avg_perdida = perdida.rolling(window=window).mean()
    rs = avg_ganancia / avg_perdida
    return 100 - (100 / (1 + rs))

# 3. Red Neuronal Engulfing (Patrón envolvente)
def es_engulfing(data):
    alcista = (data['Close'].shift(1) < data['Open'].shift(1)) & \
              (data['Open'] < data['Close'].shift(1)) & \
              (data['Close'] > data['Open'].shift(1))
    
    bajista = (data['Close'].shift(1) > data['Open'].shift(1)) & \
              (data['Open'] > data['Close'].shift(1)) & \
              (data['Close'] < data['Open'].shift(1))
    
    return alcista, bajista

# 4. Red Neuronal Doji (Patrón de indecisión)
def es_doji(data):
    doji = abs(data['Close'] - data['Open']) < (data['High'] - data['Low']) * 0.1
    return doji

# 5. Red Neuronal Estrella Fugaz
def es_estrella_fugaz(data):
    estrella_fugaz = (data['High'] - data['Open']) > 2 * (data['Open'] - data['Close'])
    return estrella_fugaz

# 6. Red Neuronal Martillo
def es_martillo(data):
    martillo = (data['Open'] - data['Low']) > 2 * (data['Close'] - data['Open'])
    return martillo

# 7. Red Neuronal Swing (Patrón de reversión)
def es_swing(data):
    swing = (data['High'].shift(1) > data['High']) & (data['Low'].shift(1) < data['Low'])
    return swing

# 8. Red Neuronal Scalping (Pequeños movimientos rápidos)
def calcular_scalping(data):
    scalping = (data['Close'] - data['Open']) / data['Open'] * 100
    return scalping

# 9. Red Neuronal Aroon, Parabolic SAR, ADX
def calcular_aroon(data, window=25):
    rolling_high = data['High'].rolling(window=window).apply(lambda x: np.argmax(x), raw=False)
    rolling_low = data['Low'].rolling(window=window).apply(lambda x: np.argmin(x), raw=False)
    aroon_up = (window - rolling_high) / window * 100
    aroon_down = (window - rolling_low) / window * 100
    return aroon_up, aroon_down

def calcular_parabolic_sar(data):
    # Implementar lógica real del Parabolic SAR
    data['PSAR'] = data['Close']  # Ejemplo temporal
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

# 10. Red Neuronal Indicadores de Bill Williams (AO, AC)
def calcular_ao_ac(data):
    ao = data['Close'].rolling(window=5).mean() - data['Close'].rolling(window=34).mean()
    ac = ao.rolling(window=5).mean() - ao.rolling(window=34).mean()
    return ao, ac

# 11. Red Neuronal Ichimoku
def calcular_ichimoku(data):
    data['tenkan_sen'] = (data['High'].rolling(window=9).max() + data['Low'].rolling(window=9).min()) / 2
    data['kijun_sen'] = (data['High'].rolling(window=26).max() + data['Low'].rolling(window=26).min()) / 2
    data['senkou_span_a'] = (data['tenkan_sen'] + data['kijun_sen']) / 2
    data['senkou_span_b'] = (data['High'].rolling(window=52).max() + data['Low'].rolling(window=52).min()) / 2
    return data['tenkan_sen'], data['kijun_sen'], data['senkou_span_a'], data['senkou_span_b']

# 12. Red Neuronal Fibonacci
def calcular_niveles_fibonacci(min_price, max_price):
    diferencia = max_price - min_price
    niveles = {
        'Nivel 23.6%': max_price - 0.236 * diferencia,
        'Nivel 38.2%': max_price - 0.382 * diferencia,
        'Nivel 50%': max_price - 0.5 * diferencia,
        'Nivel 61.8%': max_price - 0.618 * diferencia,
        'Nivel 100%': min_price,
    }
    return niveles

# Simulación de mercado (sustituir con datos reales)
data = pd.DataFrame({
    'Open': np.random.randint(100, 200, size=100),
    'Close': np.random.randint(100, 200, size=100),
    'High': np.random.randint(100, 200, size=100),
    'Low': np.random.randint(100, 200, size=100),
})

# Calcular las salidas de todas las subredes neuronales
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

# Variable objetivo
target = data['Close'].shift(-1).fillna(0)

# Escalar las características
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, shuffle=False)

# Definir la red neuronal madre
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(features.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Predicción del precio de cierre
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluar el modelo
loss = model.evaluate(X_test, y_test)
print(f"Pérdida en el conjunto de prueba: {loss}")
