import schedule
import time
from base_de_datos import obtener_datos_tiempo_real  # Importar función para obtener datos en tiempo real
from predicciones import predecir_precio
from modelo_creacion.crear_modelo import entrenar_modelo
import pandas as pd

# Lista de divisas disponibles
divisas_disponibles = [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD',
    'NZD/USD', 'CHF/JPY', 'USD/CHF', 'AUD/NZD', 'GBP/JPY',
    'EUR/GBP', 'USD/SGD', 'USD/HKD', 'AUD/JPY', 'EUR/AUD',
    'EUR/CHF', 'EUR/CAD', 'GBP/CAD', 'AUD/CAD', 'NZD/JPY',
    'GBP/NZD'
]

def main():
    print("Bienvenido al sistema de predicción de Forex.")
    api_key = "TU_API_KEY_AQUI"  # Asegúrate de insertar tu API key

    # Cargar modelo previamente entrenado
    modelo = cargar_modelo("modelos_guardados/modelo.h5")  # Asegúrate de que el modelo está en la carpeta correcta

    while True:
        print("\nOpciones:")
        print("1. Generar datos de mercado")
        print("2. Realizar predicción")
        print("3. Salir")
        opcion = input("Seleccione una opción: ")

        if opcion == '1':
            print("Generando datos de mercado...")
            # Llama a la función generadora aquí
            # generar_datos_mercado()

        elif opcion == '2':
            print("Selecciona las divisas para predecir:")
            for i, divisa in enumerate(divisas_disponibles, start=1):
                print(f"{i}. {divisa}")
                
            cantidad_divisas = int(input("¿Cuántas divisas deseas seleccionar? (Máximo 5): "))
            seleccionadas = []
            
            for _ in range(min(cantidad_divisas, 5)):
                seleccion = int(input("Ingresa el número de la divisa que deseas predecir: "))
                if 1 <= seleccion <= len(divisas_disponibles):
                    seleccionadas.append(divisas_disponibles[seleccion - 1])
                else:
                    print("Selección no válida. Intenta de nuevo.")
                    break

            if len(seleccionadas) > 0:
                for simbolo in seleccionadas:
                    print(f"Realizando predicción para {simbolo}...")
                    # Obtener datos en tiempo real
                    datos_tiempo_real = obtener_datos_tiempo_real(simbolo, api_key)
                    # Usar los últimos 60 minutos de datos para la predicción
                    datos_entrada = datos_tiempo_real[['Apertura', 'Máximo', 'Mínimo', 'Cierre', 'Volumen']].tail(60)

                    # Realizar la predicción
                    precio_predicho = predecir_precio(modelo, datos_entrada.values)
                    print(f"Precio predicho para {simbolo}: {precio_predicho[-1][0]}")  # Mostramos solo el último valor de la predicción

        elif opcion == '3':
            print("Saliendo del programa.")
            break

        else:
            print("Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    # Suponiendo que tienes tus datos en un DataFrame llamado 'datos'
    datos = pd.read_csv('ruta/a/tu/dataset.csv')  # Carga tus datos aquí

    modelo, X_test, y_test, scaler = entrenar_modelo(datos)
    
    # Aquí puedes guardar el modelo entrenado en la carpeta 'modelos_guardados'
    modelo.save('modelos_guardados/modelo.h5')  # Guarda el modelo entrenado
    main()
