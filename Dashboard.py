# Dashboard.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import json
import shap
import numpy as np
from datetime import datetime

LOG_FILE = "log.txt"

app = Flask(__name__)
CORS(app)

# Cargar modelo y columnas esperadas
try:
    model = joblib.load('random_forest_model.pkl')
    columnas_esperadas = list(model.feature_names_in_)
    
    # Inicializar Explainer
    try:
        # check_additivity=False evita errores si el modelo es complejo
        explainer = shap.TreeExplainer(model)
        print("✅ Motor de Explicabilidad (SHAP) iniciado correctamente")
    except Exception as e:
        print(f"⚠️ Advertencia: No se pudo iniciar SHAP: {e}")
        explainer = None

    print("✅ Modelo cargado correctamente")
    print(f"🧠 Columnas esperadas: {columnas_esperadas}")
except Exception as e:
    print(f"❌ Error crítico al cargar el modelo: {e}")
    model = None
    explainer = None
    columnas_esperadas = []

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no cargado'}), 500

    try:
        data = request.get_json()

        # Validación
        missing = [col for col in columnas_esperadas if col not in data]
        if missing:
            return jsonify({'error': 'Faltan variables', 'missing': missing}), 400

        # Crear DataFrame
        try:
            input_df = pd.DataFrame([{col: float(data[col]) for col in columnas_esperadas}])
        except ValueError:
            return jsonify({'error': 'Valores deben ser numéricos'}), 400

        # 1. PREDICCIÓN
        prediction_array = model.predict(input_df)
        pred = int(prediction_array[0]) 
        
        prob_array = model.predict_proba(input_df)
        probabilidad = prob_array[0].tolist()

        # 2. EXPLICACIÓN (CORRECCIÓN APLICADA AQUÍ)
        explicacion = []
        if explainer:
            try:
                # Calculamos SHAP
                # check_additivity=False hace que sea más tolerante a errores de precisión
                shap_values = explainer.shap_values(input_df, check_additivity=False)
                
                vals = None
                
                # CASO 1: Es una lista (Clasificador Binario suele devolver [Matriz_Clase0, Matriz_Clase1])
                if isinstance(shap_values, list):
                    # Tomamos la clase 1 (Positiva/Apto) o la clase 0 si solo hay una
                    idx = 1 if len(shap_values) > 1 else 0
                    vals = shap_values[idx]
                else:
                    vals = shap_values

                # CASO 2: Asegurar que sea array de numpy
                vals = np.array(vals)

                # CASO 3: APLANAR (FLATTEN) - LA SOLUCIÓN AL ERROR
                # Si viene como [[0.1, 0.2, ...]] (1 fila, N columnas), flatten lo vuelve [0.1, 0.2, ...]
                # Esto elimina el error "only length-1 arrays..."
                vals = vals.flatten()

                # Construir la lista de razones
                for i, col_name in enumerate(columnas_esperadas):
                    # Ahora 'vals[i]' es seguro un número flotante, no un array
                    impacto = float(vals[i]) 
                    
                    explicacion.append({
                        "variable": col_name,
                        "valor_input": float(input_df.iloc[0, i]),
                        "impacto": impacto,
                        "importancia_abs": abs(impacto)
                    })
                
                # Ordenar Top 5
                explicacion.sort(key=lambda x: x['importancia_abs'], reverse=True)
                explicacion = explicacion[:5]

            except Exception as e_shap:
                print(f"⚠️ Error al generar explicación SHAP: {e_shap}")
                import traceback
                traceback.print_exc() # Imprime el error completo en consola para depurar
                explicacion = []

        resultado = {
            'prediction': pred,
            'probabilidad': probabilidad,
            'interpretation': 'Apto' if pred == 1 else 'No Apto',
            'explicacion': explicacion
        }

        # Log
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input": data,
            "output": resultado
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return jsonify(resultado)

    except Exception as e:
        print(f"❌ Error general en predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/log', methods=['GET'])
def get_log():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return jsonify([json.loads(line) for line in f if line.strip()])
    except FileNotFoundError:
        return jsonify([])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)