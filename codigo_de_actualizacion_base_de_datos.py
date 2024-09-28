import requests
import psycopg2
from datetime import datetime
import time

# Conexión a la base de datos PostgreSQL
conn = psycopg2.connect(
    dbname="nombre_de_tu_base_de_datos",
    user="tu_usuario",
    password="tu_password",
    host="localhost"
)

cursor = conn.cursor()

# API Key y lista de divisas
api_key = 'TU_API_KEY'
currency_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 
                  'NZD/USD', 'CHF/JPY', 'USD/CHF', 'AUD/NZD', 'GBP/JPY', 
                  'EUR/GBP', 'USD/SGD', 'USD/HKD', 'AUD/JPY', 'EUR/AUD', 
                  'EUR/CHF', 'EUR/CAD', 'GBP/CAD', 'AUD/CAD', 'NZD/JPY', 
                  'GBP/NZD']

# Función para obtener la tasa de cambio de una API con datos adicionales
def get_exchange_rate_data(from_currency, to_currency):
    url = f'https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={from_currency}&to_symbol={to_currency}&interval=1min&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    if 'Time Series FX (1min)' in data:
        latest_time = list(data['Time Series FX (1min)'].keys())[0]
        latest_data = data['Time Series FX (1min)'][latest_time]
        return {
            'rate': float(latest_data['4. close']),
            'open': float(latest_data['1. open']),
            'high': float(latest_data['2. high']),
            'low': float(latest_data['3. low']),
            'volume': 0,  # La API Alpha Vantage no proporciona volumen, pero otras sí lo hacen
            'last_updated': latest_time
        }
    else:
        return None

# Actualización periódica de las tasas de cambio en la base de datos cada 60 segundos
while True:
    for pair in currency_pairs:
        from_currency, to_currency = pair.split('/')
        try:
            rate_data = get_exchange_rate_data(from_currency, to_currency)
            if rate_data:
                now = datetime.now()

                # Obtener el id del par de divisas
                cursor.execute("SELECT id FROM currency_pairs WHERE pair = %s", (pair,))
                pair_id = cursor.fetchone()[0]

                cursor.execute("""
                    INSERT INTO exchange_rates_history (pair_id, rate, open, high, low, volume, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                """, (pair_id, rate_data['rate'], rate_data['open'], rate_data['high'], rate_data['low'], rate_data['volume'], now))
                
                conn.commit()

        except Exception as e:
            print(f"Error al obtener la tasa para {pair}: {e}")
    
    # Pausa de 60 segundos
    time.sleep(60)

# Cerrar la conexión a la base de datos al finalizar el ciclo
cursor.close()
conn.close()
