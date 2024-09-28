import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout  
import numpy as np
import pandas as pd
import talib as ta
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt





                                             #RED NEURONAL FIBONACCI 


# Función para calcular los niveles de retroceso de Fibonacci
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

# Función para calcular extensiones de Fibonacci
def calcular_extensiones_fibonacci(min_price, max_price):
    diferencia = max_price - min_price
    extensiones = {
        'Extensión 127.2%': max_price + 0.272 * diferencia,
        'Extensión 161.8%': max_price + 0.618 * diferencia,
        'Extensión 261.8%': max_price + 1.618 * diferencia,
    }
    return extensiones

# Cargar datos ficticios de ejemplo (simulación de precios)
df = pd.DataFrame({
    'High': np.random.randint(100, 200, size=200),
    'Low': np.random.randint(50, 100, size=200),
    'Close': np.random.randint(75, 150, size=200)
})

# Calcular niveles y extensiones de Fibonacci
df['max_price'] = df['High'].rolling(window=14).max()
df['min_price'] = df['Low'].rolling(window=14).min()

df['Nivel 23.6%'] = df.apply(lambda row: calcular_niveles_fibonacci(row['min_price'], row['max_price'])['Nivel 23.6%'], axis=1)
df['Nivel 38.2%'] = df.apply(lambda row: calcular_niveles_fibonacci(row['min_price'], row['max_price'])['Nivel 38.2%'], axis=1)
df['Nivel 50%'] = df.apply(lambda row: calcular_niveles_fibonacci(row['min_price'], row['max_price'])['Nivel 50%'], axis=1)
df['Nivel 61.8%'] = df.apply(lambda row: calcular_niveles_fibonacci(row['min_price'], row['max_price'])['Nivel 61.8%'], axis=1)

df['Extensión 127.2%'] = df.apply(lambda row: calcular_extensiones_fibonacci(row['min_price'], row['max_price'])['Extensión 127.2%'], axis=1)
df['Extensión 161.8%'] = df.apply(lambda row: calcular_extensiones_fibonacci(row['min_price'], row['max_price'])['Extensión 161.8%'], axis=1)
df['Extensión 261.8%'] = df.apply(lambda row: calcular_extensiones_fibonacci(row['min_price'], row['max_price'])['Extensión 261.8%'], axis=1)

# Eliminar filas con NaN
df.dropna(inplace=True)

# Paso 1: Crear etiquetas para predicción de compra/venta
def create_labels(df, future_period=1, risk_threshold=0.01):
    df['future_close'] = df['Close'].shift(-future_period)
    df['target'] = np.where((df['future_close'] / df['Close'] - 1) > risk_threshold, 1, 0)
    df.dropna(inplace=True)
    return df

df = create_labels(df)

# Paso 2: Definir variables predictoras y objetivo
features = ['Nivel 23.6%', 'Nivel 38.2%', 'Nivel 50%', 'Nivel 61.8%', 'Extensión 127.2%', 'Extensión 161.8%', 'Extensión 261.8%']
X = df[features].values
y = df['target'].values

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Paso 3: Crear el modelo de red neuronal con capas adicionales y dropout
def create_model(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))  # Dropout para evitar sobreajuste
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Salida binaria: 1 (compra), 0 (venta)
    
    # Compilar el modelo con tasa de aprendizaje ajustable
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Paso 4: Implementar K-Fold Cross-Validation para una evaluación más robusta
def train_and_evaluate_with_kfold(X, y, n_splits=5, epochs=50, batch_size=16, learning_rate=0.001):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []
    
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Crear y entrenar el modelo
        model = create_model(X_train.shape[1], learning_rate)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
        
        # Evaluar el modelo
        predictions = (model.predict(X_test) > 0.5).astype("int32")
        accuracy = accuracy_score(y_test, predictions)
        accuracy_scores.append(accuracy)
    
    avg_accuracy = np.mean(accuracy_scores)
    print(f"Average K-Fold Accuracy: {avg_accuracy * 100:.2f}%")
    
    return avg_accuracy, history

# Entrenar y evaluar el modelo con validación cruzada
avg_accuracy, history = train_and_evaluate_with_kfold(X_scaled, y, n_splits=5, epochs=50, batch_size=16, learning_rate=0.0005)

# Paso 5: Visualización del historial de entrenamiento
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # Gráfico de pérdida (loss)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfico de precisión (accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Paso 6: Realizar predicciones con datos nuevos
nuevos_datos = np.array([[130, 120, 110, 100, 150, 180, 220]])  # Ejemplo con niveles y extensiones de Fibonacci
nuevos_datos_scaled = scaler.transform(nuevos_datos)  # Normalizar los nuevos datos
prediccion = model.predict(nuevos_datos_scaled)
print(f"Predicción (0 = baja, 1 = sube): {prediccion}")







                                                  #RED NEURONAL ICHIMOKU


# Paso 1: Cálculo de los componentes del indicador Ichimoku
def calcular_ichimoku(df):
    # Línea de conversión (Tenkan-Sen)
    df['Tenkan-Sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    
    # Línea base (Kijun-Sen)
    df['Kijun-Sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    
    # Senkou Span A (Primer borde de la nube)
    df['Senkou Span A'] = ((df['Tenkan-Sen'] + df['Kijun-Sen']) / 2).shift(26)
    
    # Senkou Span B (Segundo borde de la nube)
    df['Senkou Span B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
    
    # Chikou Span (Línea de retraso)
    df['Chikou Span'] = df['Close'].shift(-26)

    return df

# Cargar datos ficticios de ejemplo (simulación de precios)
df = pd.DataFrame({
    'High': np.random.randint(100, 200, size=200),
    'Low': np.random.randint(50, 100, size=200),
    'Close': np.random.randint(75, 150, size=200)
})

# Calcular los componentes de Ichimoku
df = calcular_ichimoku(df)

# Eliminar filas con NaN resultantes del cálculo de los componentes de Ichimoku
df.dropna(inplace=True)

# Paso 2: Crear nuevas características y definir la variable objetivo
def create_labels(df, future_period=1, risk_threshold=0.01):
    df['future_close'] = df['Close'].shift(-future_period)
    df['target'] = np.where((df['future_close'] / df['Close'] - 1) > risk_threshold, 1, 0)
    df.dropna(inplace=True)
    return df

df = create_labels(df)

# Paso 3: Definir variables predictoras (features) y objetivo (target)
features = ['Tenkan-Sen', 'Kijun-Sen', 'Senkou Span A', 'Senkou Span B', 'Chikou Span']
X = df[features].values
y = df['target'].values

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Paso 4: Crear el modelo de red neuronal con capas adicionales y dropout
def create_model(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))  # Dropout para evitar sobreajuste
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Salida binaria: 1 (compra), 0 (venta)
    
    # Compilar el modelo con tasa de aprendizaje ajustable
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Paso 5: Implementar K-Fold Cross-Validation para una evaluación más robusta
def train_and_evaluate_with_kfold(X, y, n_splits=5, epochs=50, batch_size=16, learning_rate=0.001):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []
    
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Crear y entrenar el modelo
        model = create_model(X_train.shape[1], learning_rate)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
        
        # Evaluar el modelo
        predictions = (model.predict(X_test) > 0.5).astype("int32")
        accuracy = accuracy_score(y_test, predictions)
        accuracy_scores.append(accuracy)
    
    avg_accuracy = np.mean(accuracy_scores)
    print(f"Average K-Fold Accuracy: {avg_accuracy * 100:.2f}%")
    
    return avg_accuracy, history

# Entrenar y evaluar el modelo con validación cruzada
avg_accuracy, history = train_and_evaluate_with_kfold(X_scaled, y, n_splits=5, epochs=50, batch_size=16, learning_rate=0.0005)

# Paso 6: Visualización del historial de entrenamiento
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # Gráfico de pérdida (loss)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfico de precisión (accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Paso 7: Realizar predicciones con datos nuevos
nuevos_datos = np.array([[130, 120, 110, 100, 105]])  # Ejemplo de datos con los componentes de Ichimoku
nuevos_datos_scaled = scaler.transform(nuevos_datos)  # Normalizar los nuevos datos
prediccion = model.predict(nuevos_datos_scaled)
print(f"Predicción (0 = baja, 1 = sube): {prediccion}")









                                                     #RED NEURONAL INDICADORES DE BILLS WILLIANS

# Paso 1: Implementar los indicadores de Bill Williams
# Función para calcular el indicador Alligator (Jaw, Teeth, Lips)
def calcular_alligator(df):
    df['Jaw'] = df['Close'].rolling(window=13).mean().shift(8)
    df['Teeth'] = df['Close'].rolling(window=8).mean().shift(5)
    df['Lips'] = df['Close'].rolling(window=5).mean().shift(3)
    return df

# Función para calcular fractales (high/lows fractals)
def calcular_fractales(df):
    df['Fractal Alcista'] = df['High'][(df['High'].shift(2) < df['High']) & (df['High'].shift(1) < df['High']) & (df['High'] > df['High'].shift(-1)) & (df['High'] > df['High'].shift(-2))]
    df['Fractal Bajista'] = df['Low'][(df['Low'].shift(2) > df['Low']) & (df['Low'].shift(1) > df['Low']) & (df['Low'] < df['Low'].shift(-1)) & (df['Low'] < df['Low'].shift(-2))]
    return df

# Función para calcular el Awesome Oscillator (AO)
def calcular_ao(df):
    df['AO'] = df['Close'].rolling(window=5).mean() - df['Close'].rolling(window=34).mean()
    return df

# Función para calcular el Accelerator Oscillator (AC)
def calcular_ac(df):
    df['AC'] = df['AO'] - df['AO'].rolling(window=5).mean()
    return df

# Función para calcular el Gator Oscillator
def calcular_gator(df):
    df['Gator Superior'] = abs(df['Jaw'] - df['Teeth'])
    df['Gator Inferior'] = abs(df['Teeth'] - df['Lips'])
    return df

# Cargar datos ficticios de ejemplo (simulación de precios)
df = pd.DataFrame({
    'High': np.random.randint(100, 200, size=100),
    'Low': np.random.randint(50, 100, size=100),
    'Close': np.random.randint(75, 150, size=100)
})

# Calcular todos los indicadores de Bill Williams
df = calcular_alligator(df)
df = calcular_fractales(df)
df = calcular_ao(df)
df = calcular_ac(df)
df = calcular_gator(df)

# Eliminar filas con NaN resultantes del cálculo de los indicadores
df.dropna(inplace=True)

# Paso 2: Crear nuevas características y definir la variable objetivo
def create_labels(df, future_period=1, risk_threshold=0.01):
    df['future_close'] = df['Close'].shift(-future_period)
    df['target'] = np.where((df['future_close'] / df['Close'] - 1) > risk_threshold, 1, 0)
    df.dropna(inplace=True)
    return df

df = create_labels(df)

# Definir las variables predictoras (features) y objetivo (target)
features = ['Jaw', 'Teeth', 'Lips', 'Fractal Alcista', 'Fractal Bajista', 'AO', 'AC', 'Gator Superior', 'Gator Inferior']
X = df[features].values
y = df['target'].values

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Paso 3: Crear el modelo de red neuronal con capas adicionales
def create_model(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))  # Dropout para evitar sobreajuste
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Salida binaria: 1 (compra), 0 (venta)
    
    # Compilar el modelo con ajuste de la tasa de aprendizaje
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Paso 4: Implementar K-Fold Cross-Validation
def train_and_evaluate_with_kfold(X, y, n_splits=5, epochs=50, batch_size=16, learning_rate=0.001):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []
    
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Crear y entrenar el modelo
        model = create_model(X_train.shape[1], learning_rate)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
        
        # Evaluar el modelo
        predictions = (model.predict(X_test) > 0.5).astype("int32")
        accuracy = accuracy_score(y_test, predictions)
        accuracy_scores.append(accuracy)
    
    avg_accuracy = np.mean(accuracy_scores)
    print(f"Average K-Fold Accuracy: {avg_accuracy * 100:.2f}%")
    
    return avg_accuracy, history

# Entrenar y evaluar el modelo con validación cruzada
avg_accuracy, history = train_and_evaluate_with_kfold(X_scaled, y, n_splits=5, epochs=50, batch_size=16, learning_rate=0.0005)

# Paso 5: Visualización del historial de entrenamiento
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # Gráfico de pérdida (loss)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfico de precisión (accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Paso 6: Realizar predicciones con datos nuevos
nuevos_datos = np.array([[130, 120, 110, 1, 0, 15, 5, 2, 1]])  # Ejemplo de datos con los indicadores
nuevos_datos_scaled = scaler.transform(nuevos_datos)  # Normalizar los nuevos datos
prediccion = model.predict(nuevos_datos_scaled)
print(f"Predicción (0 = baja, 1 = sube): {prediccion}")















                                                    #RED NEURONAL AROON, PARABOLIC SAR, ADX 



# Paso 1: Crear datos de ejemplo y calcular indicadores técnicos (Aroon, Parabolic SAR, ADX)
df = pd.DataFrame({
    'High': np.random.randint(100, 200, size=100),
    'Low': np.random.randint(50, 100, size=100),
    'Close': np.random.randint(75, 150, size=100)
})

# Calcular indicadores técnicos
df['Aroon Up'], df['Aroon Down'] = ta.AROON(df['High'], df['Low'], timeperiod=14)
df['Parabolic SAR'] = ta.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2)
df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)

# Eliminar filas con NaN resultantes del cálculo de los indicadores
df.dropna(inplace=True)

# Crear la etiqueta: 1 si el precio sube, 0 si baja (basado en ganancias/riesgos con un umbral)
def create_labels(df, future_period=1, risk_threshold=0.01):
    df['future_close'] = df['Close'].shift(-future_period)
    df['target'] = np.where((df['future_close'] / df['Close'] - 1) > risk_threshold, 1, 0)
    df.dropna(inplace=True)
    return df

df = create_labels(df)

# Features: Aroon Up, Aroon Down, Parabolic SAR, ADX
X = df[['Aroon Up', 'Aroon Down', 'Parabolic SAR', 'ADX']].values
y = df['target'].values

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Paso 2: Crear el modelo de red neuronal con capas adicionales y ajuste de tasa de aprendizaje
def create_model(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Salida binaria: 1 (compra), 0 (venta)
    
    # Ajuste de tasa de aprendizaje
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Paso 3: Implementar K-Fold Cross-Validation
def train_and_evaluate_with_kfold(X, y, n_splits=5, epochs=50, batch_size=16, learning_rate=0.001):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []
    
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Crear y entrenar el modelo
        model = create_model(X_train.shape[1], learning_rate)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
        
        # Evaluar el modelo
        predictions = (model.predict(X_test) > 0.5).astype("int32")
        accuracy = accuracy_score(y_test, predictions)
        accuracy_scores.append(accuracy)
    
    avg_accuracy = np.mean(accuracy_scores)
    print(f"Average K-Fold Accuracy: {avg_accuracy * 100:.2f}%")
    
    return avg_accuracy, history

# Paso 4: Entrenar y evaluar el modelo con validación cruzada
avg_accuracy, history = train_and_evaluate_with_kfold(X_scaled, y, n_splits=5, epochs=50, batch_size=16, learning_rate=0.0005)

# Paso 5: Visualización del historial de entrenamiento
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # Gráfico de pérdida (loss)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfico de precisión (accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Paso 6: Predicción con datos nuevos
nuevos_datos = np.array([[80, 20, 135, 25]])  # Ejemplo de Aroon Up, Aroon Down, Parabolic SAR, ADX
nuevos_datos_scaled = scaler.transform(nuevos_datos)  # Normalizar los nuevos datos
prediccion = model.predict(nuevos_datos_scaled)
print(f"Predicción (0 = baja, 1 = sube): {prediccion}")













                                                     #RED NUERONAL SCALPING 

# Paso 1: Obtener datos históricos
def get_data(symbol='BTC-USD', period='60d', interval='1m'):
    data = yf.download(tickers=symbol, period=period, interval=interval)
    data.dropna(inplace=True)
    return data

data = get_data()

# Paso 2: Función para detectar el patrón de variación porcentual diaria
def detect_patterns(data):
    data['daily_return'] = data['Close'].pct_change()  # Variación porcentual diaria
    data['SMA_20'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()
    
    bb = BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['bb_mavg'] = bb.bollinger_mavg()
    data['bb_hband'] = bb.bollinger_hband()
    data['bb_lband'] = bb.bollinger_lband()
    
    data['rsi'] = RSIIndicator(close=data['Close'], window=14).rsi()
    data.dropna(inplace=True)
    return data

# Aplicamos la función para detectar patrones
data = detect_patterns(data)

# Paso 3: Crear nuevas características (variación porcentual diaria) y ajustar las etiquetas de compra/venta
def create_features(data, future_period=5, risk_threshold=0.01):
    # Definir variables predictoras
    features = ['daily_return', 'SMA_20', 'bb_mavg', 'bb_hband', 'bb_lband', 'rsi']
    X = data[features].values

    # Definir la variable objetivo basada en ganancias/riesgos
    data['future_close'] = data['Close'].shift(-future_period)
    data['target'] = np.where((data['future_close'] / data['Close'] - 1) > risk_threshold, 1, 0)
    data.dropna(inplace=True)
    y = data['target'].values

    # Normalizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

X_scaled, y = create_features(data)

# Paso 4: Crear el modelo de red neuronal con capas adicionales y ajuste de tasa de aprendizaje
def create_model(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.4))  # Aumentamos el dropout a 40%
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Salida binaria: 1 (compra), 0 (venta)
    
    # Ajuste de tasa de aprendizaje
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Paso 5: Implementación de K-Fold Cross-Validation
def train_and_evaluate_with_kfold(X, y, n_splits=5, epochs=50, batch_size=64, learning_rate=0.001):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []
    
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Crear y entrenar el modelo
        model = create_model(X_train.shape[1], learning_rate)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
        
        # Evaluar el modelo
        predictions = (model.predict(X_test) > 0.5).astype("int32")
        accuracy = accuracy_score(y_test, predictions)
        accuracy_scores.append(accuracy)
    
    avg_accuracy = np.mean(accuracy_scores)
    print(f"Average K-Fold Accuracy: {avg_accuracy * 100:.2f}%")
    
    return avg_accuracy, history

# Entrenamos y evaluamos el modelo con validación cruzada
avg_accuracy, history = train_and_evaluate_with_kfold(X_scaled, y, n_splits=5, epochs=50, batch_size=64, learning_rate=0.0005)

# Paso 6: Visualización del historial de entrenamiento
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # Gráfico de pérdida (loss)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfico de precisión (accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)











                                                     #RED NEURONAL SWING
  



# Paso 1: Obtener datos históricos
def get_data(symbol='BTC-USD', period='60d', interval='1m'):
    data = yf.download(tickers=symbol, period=period, interval=interval)
    data.dropna(inplace=True)
    return data

data = get_data()

# Paso 2: Función para detectar el patrón de variación porcentual diaria
def detect_patterns(data):
    data['daily_return'] = data['Close'].pct_change()  # Variación porcentual diaria
    data['SMA_20'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()
    
    bb = BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['bb_mavg'] = bb.bollinger_mavg()
    data['bb_hband'] = bb.bollinger_hband()
    data['bb_lband'] = bb.bollinger_lband()
    
    data['rsi'] = RSIIndicator(close=data['Close'], window=14).rsi()
    data.dropna(inplace=True)
    return data

# Aplicamos la función para detectar patrones
data = detect_patterns(data)

# Paso 3: Crear nuevas características (variación porcentual diaria) y ajustar las etiquetas de compra/venta
def create_features(data, future_period=5, risk_threshold=0.01):
    # Definir variables predictoras
    features = ['daily_return', 'SMA_20', 'bb_mavg', 'bb_hband', 'bb_lband', 'rsi']
    X = data[features].values

    # Definir la variable objetivo basada en ganancias/riesgos
    data['future_close'] = data['Close'].shift(-future_period)
    data['target'] = np.where((data['future_close'] / data['Close'] - 1) > risk_threshold, 1, 0)
    data.dropna(inplace=True)
    y = data['target'].values

    # Normalizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

X_scaled, y = create_features(data)

# Paso 4: Crear el modelo de red neuronal con capas adicionales y ajuste de tasa de aprendizaje
def create_model(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.4))  # Aumentamos el dropout a 40%
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Salida binaria: 1 (compra), 0 (venta)
    
    # Ajuste de tasa de aprendizaje
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Paso 5: Implementación de K-Fold Cross-Validation
def train_and_evaluate_with_kfold(X, y, n_splits=5, epochs=50, batch_size=64, learning_rate=0.001):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []
    
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Crear y entrenar el modelo
        model = create_model(X_train.shape[1], learning_rate)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
        
        # Evaluar el modelo
        predictions = (model.predict(X_test) > 0.5).astype("int32")
        accuracy = accuracy_score(y_test, predictions)
        accuracy_scores.append(accuracy)
    
    avg_accuracy = np.mean(accuracy_scores)
    print(f"Average K-Fold Accuracy: {avg_accuracy * 100:.2f}%")
    
    return avg_accuracy, history

# Entrenamos y evaluamos el modelo con validación cruzada
avg_accuracy, history = train_and_evaluate_with_kfold(X_scaled, y, n_splits=5, epochs=50, batch_size=64, learning_rate=0.0005)

# Paso 6: Visualización del historial de entrenamiento
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # Gráfico de pérdida (loss)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfico de precisión (accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)









                                                                   #PATRONES DE VELAS

                                         #RED NUERONAL AMRTILLO                                                        
 
 #Función para detectar el patrón Martillo
def es_martillo(data):
    cuerpo = abs(data['Close'] - data['Open'])
    sombra_inferior = data['Low'] - min(data['Open'], data['Close'])
    sombra_superior = max(data['Open'], data['Close']) - data['High']
    es_martillo = (sombra_inferior >= 2 * cuerpo) and (sombra_superior <= cuerpo * 0.1)
    return es_martillo

# Supongamos que tenemos un DataFrame con datos históricos
datos_mercado = pd.DataFrame({
    'Open': [100, 102, 98, 95],
    'Close': [105, 100, 99, 94],
    'High': [106, 103, 101, 97],
    'Low': [96, 97, 93, 90]
})

# Aplicamos la función para detectar el patrón Martillo
datos_mercado['Martillo'] = datos_mercado.apply(es_martillo, axis=1)

# Crear nuevas características (variación porcentual diaria)
datos_mercado['% Change'] = (datos_mercado['Close'] - datos_mercado['Open']) / datos_mercado['Open']

# Definir variable objetivo (Martillo)
datos_mercado['Martillo'] = datos_mercado['Martillo'].astype(int)

# Definir variables predictoras (Open, High, Low, Close, % Change)
X = datos_mercado[['Open', 'High', 'Low', 'Close', '% Change']].values
y = datos_mercado['Martillo'].values

# Normalizar los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Función para crear el modelo de red neuronal
def crear_modelo():
    model = Sequential([
        Dense(256, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# K-fold cross-validation para evaluación más robusta
kf = KFold(n_splits=5, shuffle=True, random_state=42)
modelo_clasificador = KerasClassifier(build_fn=crear_modelo, epochs=150, batch_size=16, verbose=0)
resultados = cross_val_score(modelo_clasificador, X_scaled, y, cv=kf)
print(f"Precisión promedio con validación cruzada: {np.mean(resultados) * 100:.2f}%")

# Entrenar el modelo final
model = crear_modelo()
history = model.fit(X_train, y_train, epochs=150, batch_size=16, validation_data=(X_test, y_test))

# Evaluar el modelo 
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")

# Realizar predicciones
predicciones = (model.predict(X_test) > 0.5).astype("int32")

# Evaluar la precisión del modelo
print(classification_report(y_test, predicciones))
print(confusion_matrix(y_test, predicciones))

# Visualización del historial de entrenamiento
plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión en validación')
plt.title('Precisión del modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()










                                          #RED NEURONAL  ESTRELLA FUGAZ 


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Función para detectar el patrón de Estrella Fugaz
def es_estrella_fugaz(data):
    cuerpo = abs(data['Close'] - data['Open'])
    sombra_superior = data['High'] - max(data['Open'], data['Close'])
    sombra_inferior = min(data['Open'], data['Close']) - data['Low']
    
    # Condiciones para ser una Estrella Fugaz
    es_estrella = (sombra_superior >= 2 * cuerpo) and (sombra_inferior <= cuerpo * 0.1)
    
    return es_estrella

# Supongamos que tienes un DataFrame con datos históricos
datos_mercado = pd.DataFrame({
    'Open': [100, 102, 110, 120],
    'Close': [105, 100, 108, 119],
    'High': [106, 104, 115, 125],
    'Low': [95, 97, 109, 117]
})

# Aplicamos la función para detectar el patrón
datos_mercado['Estrella_Fugaz'] = datos_mercado.apply(es_estrella_fugaz, axis=1)

# Crear nuevas características (variación porcentual diaria)
datos_mercado['% Change'] = (datos_mercado['Close'] - datos_mercado['Open']) / datos_mercado['Open']

# Definir variable objetivo
datos_mercado['Estrella_Fugaz'] = datos_mercado['Estrella_Fugaz'].astype(int)

# Definir variables predictoras
X = datos_mercado[['Open', 'High', 'Low', 'Close', '% Change']].values
y = datos_mercado['Estrella_Fugaz'].values  # Convertir True/False a 1/0

# Normalizar los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Implementar K-Fold Cross Validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
precision_pliegues = []

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Crear el modelo de red neuronal
    model = Sequential([
        Dense(256, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)),  # Más neuronas y regularización L2
        Dropout(0.4),  # Dropout ajustado para evitar sobreajuste
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),  # Segunda capa con más neuronas y regularización
        Dropout(0.3),  # Dropout ajustado
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),  # Tercera capa
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Capa de salida con activación sigmoide
    ])

    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    history = model.fit(X_train, y_train, epochs=150, batch_size=16, verbose=0, validation_data=(X_test, y_test))

    # Evaluar el modelo en el pliegue de prueba
    predicciones = (model.predict(X_test) > 0.5).astype("int32")
    precision = accuracy_score(y_test, predicciones)
    precision_pliegues.append(precision)

    # Mostrar resultados del pliegue actual
    print(f"Precisión en el pliegue actual: {precision * 100:.2f}%")
    print(classification_report(y_test, predicciones))
    print(confusion_matrix(y_test, predicciones))

# Calcular la precisión promedio de todos los pliegues
precision_promedio = sum(precision_pliegues) / k
print(f"Precisión promedio en validación cruzada: {precision_promedio * 100:.2f}%")

# Visualización del historial de entrenamiento (solo del último pliegue)
plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión en validación')
plt.title('Precisión del modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()











                                                     #RED NEUROMAL DOJI



import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Función para detectar el patrón Doji
def es_doji(row):
    cuerpo = abs(row['Close'] - row['Open'])
    rango = row['High'] - row['Low']
    return cuerpo <= (0.1 * rango)  # Ajusta el umbral según sea necesario

# Datos ficticios (reemplaza esto con datos reales)
datos_mercado = pd.DataFrame({
    'Open': [100, 98, 105, 110, 108, 109, 107],
    'Close': [98, 105, 109, 106, 100, 101, 102],
    'High': [105, 107, 110, 115, 112, 111, 109],
    'Low': [95, 96, 102, 105, 99, 100, 98]
})

# Aplicamos la función para detectar el Doji
datos_mercado['Doji'] = datos_mercado.apply(es_doji, axis=1)

# Crear nuevas características (variación porcentual diaria)
datos_mercado['% Change'] = (datos_mercado['Close'] - datos_mercado['Open']) / datos_mercado['Open']

# Definir variable objetivo (Ejemplo: Doji como 1 o 0)
datos_mercado['Estrella_Fugaz'] = datos_mercado['Doji'].astype(int)

# Definir variables predictoras (X) y objetivo (y)
X = datos_mercado[['Open', 'High', 'Low', 'Close', '% Change']].fillna(0).values
y = datos_mercado['Estrella_Fugaz'].values  # Convertir True/False a 1/0

# Normalizar los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Implementar K-Fold Cross Validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
precision_pliegues = []

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Crear el modelo de red neuronal
    model = Sequential([
        Dense(128, input_dim=X_train.shape[1], activation='relu'),  # Capa oculta con 128 neuronas
        Dropout(0.3),  # Dropout para evitar el sobreajuste
        Dense(64, activation='relu'),  # Capa oculta con 64 neuronas
        Dropout(0.2),  # Dropout adicional
        Dense(1, activation='sigmoid')  # Capa de salida para clasificación binaria
    ])

    # Compilar el modelo
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    history = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)

    # Evaluar el modelo en el pliegue de prueba
    predicciones = (model.predict(X_test) > 0.5).astype("int32")
    precision = accuracy_score(y_test, predicciones)
    precision_pliegues.append(precision)
    
    # Mostrar resultados del pliegue actual
    print(f"Precisión en el pliegue actual: {precision * 100:.2f}%")
    print(classification_report(y_test, predicciones))

# Calcular la precisión promedio de todos los pliegues
precision_promedio = sum(precision_pliegues) / k
print(f"Precisión promedio en validación cruzada: {precision_promedio * 100:.2f}%")

# Visualización del historial de entrenamiento (solo del último pliegue)
plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión en validación')
plt.title('Precisión del modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()











                                                     #RED NEURONAL NEGULFIN



import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Función para calcular RSI (opcional si lo necesitas)
def calcular_rsi(data, window=14):
    delta = data['Close'].diff()
    ganancia = delta.where(delta > 0, 0)
    perdida = -delta.where(delta < 0, 0)
    avg_ganancia = ganancia.rolling(window=window).mean()
    avg_perdida = perdida.rolling(window=window).mean()
    rs = avg_ganancia / avg_perdida
    return 100 - (100 / (1 + rs))

# Función para detectar patrón Engulfing (opcional si lo necesitas)
def es_engulfing(data):
    alcista = (data['Close'].shift(1) < data['Open'].shift(1)) & \
              (data['Open'] < data['Close'].shift(1)) & \
              (data['Close'] > data['Open'].shift(1))
    
    bajista = (data['Close'].shift(1) > data['Open'].shift(1)) & \
              (data['Open'] > data['Close'].shift(1)) & \
              (data['Close'] < data['Open'].shift(1))
    
    return alcista, bajista

# Datos ficticios (reemplaza esto con datos reales)
datos_mercado = pd.DataFrame({
    'Open': [100, 98, 105, 110, 108, 109, 107],
    'Close': [98, 105, 109, 106, 100, 101, 102],
    'High': [105, 107, 110, 115, 112, 111, 109],
    'Low': [95, 96, 102, 105, 99, 100, 98]
})

# Detectar patrones Engulfing
datos_mercado['Engulfing_Alcista'], datos_mercado['Engulfing_Bajista'] = es_engulfing(datos_mercado)

# Crear nuevas características (variación porcentual diaria)
datos_mercado['Var_Pct'] = datos_mercado['Close'].pct_change() * 100

# Definir variables predictoras (X) y objetivo (y)
X = datos_mercado[['Open', 'High', 'Low', 'Close', 'Var_Pct']].fillna(0).values
y = datos_mercado['Engulfing_Alcista'].astype(int).values  # Convertir True/False a 1/0

# Normalizar los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Número de pliegues para la validación cruzada
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Lista para guardar las precisiones en cada pliegue
precision_pliegues = []

# Realizar validación cruzada
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Crear el modelo de red neuronal
    model = Sequential([
        Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(1, activation='sigmoid')  # Capa de salida para clasificación binaria
    ])

    # Compilar el modelo
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

    # Evaluar el modelo en el pliegue de prueba
    predicciones = (model.predict(X_test) > 0.5).astype("int32")
    precision = accuracy_score(y_test, predicciones)
    
    # Guardar la precisión del pliegue actual
    precision_pliegues.append(precision)

    # Evaluar la precisión del modelo en cada pliegue
    print(f"Precisión en el pliegue actual: {precision * 100:.2f}%")
    print(classification_report(y_test, predicciones))

# Calcular la precisión promedio de todos los pliegues
precision_promedio = sum(precision_pliegues) / k
print(f"Precisión promedio en validación cruzada: {precision_promedio * 100:.2f}%")

# Visualización del historial de entrenamiento
plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión en validación')
plt.title('Precisión del modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()








                                              






                                                    # RED NEURONAL INDICE DE FUERZA RELATIVA


# Función para calcular el RSI
def calcular_rsi(data, window=14):
    delta = data['Close'].diff()
    ganancia = delta.where(delta > 0, 0)
    perdida = -delta.where(delta < 0, 0)
    avg_ganancia = ganancia.rolling(window=window).mean()
    avg_perdida = perdida.rolling(window=window).mean()
    rs = avg_ganancia / avg_perdida
    return 100 - (100 / (1 + rs))

# Función para detectar patrón Engulfing
def es_engulfing(data):
    alcista = (data['Close'].shift(1) < data['Open'].shift(1)) & \
              (data['Open'] < data['Close'].shift(1)) & \
              (data['Close'] > data['Open'].shift(1))
    
    bajista = (data['Close'].shift(1) > data['Open'].shift(1)) & \
              (data['Open'] > data['Close'].shift(1)) & \
              (data['Close'] < data['Open'].shift(1))
    
    return alcista, bajista

# Datos ficticios (puedes reemplazarlo con datos reales)
datos_mercado = pd.DataFrame({
    'Open': [100, 98, 105, 110, 108, 112, 120],
    'Close': [98, 105, 109, 106, 100, 114, 118],
    'High': [105, 107, 110, 115, 112, 116, 122],
    'Low': [95, 96, 102, 105, 99, 110, 115]
})

# Calcular el RSI
datos_mercado['RSI'] = calcular_rsi(datos_mercado)

# Detectar patrones Engulfing
datos_mercado['Engulfing_Alcista'], datos_mercado['Engulfing_Bajista'] = es_engulfing(datos_mercado)

# Paso 1: Preprocesar los datos
# Generar etiquetas para compra (1) y venta (0)
datos_mercado['Target'] = np.where(datos_mercado['Close'].shift(-1) > datos_mercado['Close'], 1, 0)
datos_mercado.dropna(inplace=True)  # Eliminar filas con valores NaN

# Definir las características (features) y la etiqueta (target)
X = datos_mercado[['RSI', 'Engulfing_Alcista', 'Engulfing_Bajista']].astype(int).values
y = datos_mercado['Target'].values

# Normalizar los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Paso 2: Crear la red neuronal con capas adicionales y dropout
def create_model(input_dim, learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_dim=input_dim, activation='relu'),  # Capa oculta con 128 neuronas
        tf.keras.layers.Dropout(0.3),  # Dropout para evitar el sobreajuste
        tf.keras.layers.Dense(64, activation='relu'),  # Capa oculta con 64 neuronas
        tf.keras.layers.Dropout(0.2),  # Dropout adicional
        tf.keras.layers.Dense(1, activation='sigmoid')  # Capa de salida con activación sigmoide para clasificación binaria
    ])
    
    # Compilar el modelo
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Paso 3: Implementar K-Fold Cross-Validation para evaluación más robusta
def train_and_evaluate_with_kfold(X, y, n_splits=5, epochs=50, batch_size=16, learning_rate=0.001):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []
    
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Crear y entrenar el modelo
        model = create_model(X_train.shape[1], learning_rate)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
        
        # Evaluar el modelo
        predictions = (model.predict(X_test) > 0.5).astype("int32")
        accuracy = accuracy_score(y_test, predictions)
        accuracy_scores.append(accuracy)
    
    avg_accuracy = np.mean(accuracy_scores)
    print(f"Average K-Fold Accuracy: {avg_accuracy * 100:.2f}%")
    
    return avg_accuracy, history

# Entrenar y evaluar el modelo con validación cruzada
avg_accuracy, history = train_and_evaluate_with_kfold(X_scaled, y, n_splits=5, epochs=50, batch_size=16, learning_rate=0.0005)

# Paso 4: Visualización del historial de entrenamiento
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # Gráfico de pérdida (loss)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfico de precisión (accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Paso 5: Realizar predicciones
predicciones = (model.predict(X_scaled[:2]) > 0.5).astype("int32")
print(f'Predicciones para los primeros dos ejemplos: {predicciones.flatten()}')








                                            #RED NEUORONAL MACD




# Función para calcular el MACD
def calcular_macd(data, short_window=12, long_window=26, signal_window=9):
    ema_short = data['Close'].ewm(span=short_window, adjust=False).mean()
    ema_long = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    histograma = macd - signal
    return macd, signal, histograma

# Función para detectar patrón Engulfing
def es_engulfing(data):
    alcista = (data['Close'].shift(1) < data['Open'].shift(1)) & \
              (data['Open'] < data['Close'].shift(1)) & \
              (data['Close'] > data['Open'].shift(1))
    
    bajista = (data['Close'].shift(1) > data['Open'].shift(1)) & \
              (data['Open'] > data['Close'].shift(1)) & \
              (data['Close'] < data['Open'].shift(1))
    
    return alcista, bajista

# Datos ficticios de ejemplo
datos_mercado = pd.DataFrame({
    'Open': [100, 98, 105, 110, 108, 115, 120],
    'Close': [98, 105, 109, 106, 100, 117, 119],
    'High': [105, 107, 110, 115, 112, 118, 122],
    'Low': [95, 96, 102, 105, 99, 112, 116]
})

# Calcular el MACD
datos_mercado['MACD'], datos_mercado['Signal'], _ = calcular_macd(datos_mercado)

# Detectar patrones Engulfing
datos_mercado['Engulfing_Alcista'], datos_mercado['Engulfing_Bajista'] = es_engulfing(datos_mercado)

# Crear etiqueta de salida: 1 si hay Engulfing Alcista, 0 si hay Engulfing Bajista
datos_mercado['Target'] = np.where(datos_mercado['Engulfing_Alcista'], 1, 
                                    np.where(datos_mercado['Engulfing_Bajista'], 0, np.nan))
datos_mercado.dropna(inplace=True)

# Variables predictoras
X = datos_mercado[['MACD', 'Signal']].values
y = datos_mercado['Target'].values

# Normalizar los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Paso 1: Ajuste de hiperparámetros y capas adicionales con dropout
def create_model(input_dim, learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_dim=input_dim, activation='relu'),
        tf.keras.layers.Dropout(0.3),  # Evitar sobreajuste
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Salida binaria
    ])
    
    # Compilar el modelo
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Paso 2: Implementar K-Fold Cross-Validation para una evaluación más robusta
def train_and_evaluate_with_kfold(X, y, n_splits=5, epochs=50, batch_size=16, learning_rate=0.001):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []
    
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Crear y entrenar el modelo
        model = create_model(X_train.shape[1], learning_rate)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
        
        # Evaluar el modelo
        predictions = (model.predict(X_test) > 0.5).astype("int32")
        accuracy = accuracy_score(y_test, predictions)
        accuracy_scores.append(accuracy)
    
    avg_accuracy = np.mean(accuracy_scores)
    print(f"Average K-Fold Accuracy: {avg_accuracy * 100:.2f}%")
    
    return avg_accuracy, history

# Paso 3: Entrenar y evaluar el modelo con validación cruzada
avg_accuracy, history = train_and_evaluate_with_kfold(X_scaled, y, n_splits=5, epochs=50, batch_size=16, learning_rate=0.0005)

# Paso 4: Visualización del historial de entrenamiento
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # Gráfico de pérdida (loss)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfico de precisión (accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Paso 5: Realizar predicciones con los datos de prueba
prediccion = model.predict(X_scaled[:2])
print(f'Predicciones para los primeros dos ejemplos: {prediccion}')

