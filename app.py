import streamlit as st
import pandas as pd
import pymongo
from datetime import datetime
from google import genai
import os

# =======================
# CONFIGURACIÓN
# =======================
st.set_page_config(page_title="ML Data Describer", page_icon="📊", layout="wide")
st.title("📊 Data Describer para Machine Learning")

# =======================
# CONEXIÓN A MONGODB
# =======================
@st.cache_resource
def init_mongo_connection():
    try:
        # Intenta leer de Streamlit, si falla va directo al entorno del sistema
        try:
            uri = st.secrets["mongodb_uri"]
        except Exception:
            uri = os.environ.get("STREAMLIT_MONGODB_URI")

        # Verificación de seguridad si ambas opciones fallan
        if not uri:
            raise ValueError("No se encontró la variable MONGODB_URI en st.secrets ni en el entorno.")

        client = pymongo.MongoClient(uri)
        # Seleccionamos la base de datos y la colección
        db = client["ml_describer_db"]
        return db["analisis_historial"]
        
    except Exception as e:
        st.error(f"❌ Error al conectar con MongoDB: {e}")
        st.stop()

coleccion_historial = init_mongo_connection()

# =======================
# BARRA LATERAL (SIDEBAR)
# =======================
with st.sidebar:
    st.header("🔑 Configuración")
    
    # Intenta leer la API Key con la misma estrategia híbrida
    try:
        api_key = st.secrets["google_api_key"]
    except Exception:
        api_key = os.environ.get("STREAMLIT_GOOGLE_API_KEY")

    # Validamos si logramos rescatar la API Key
    if api_key:
        st.success("API Key e infraestructura cargadas correctamente.")
    else:
        st.error("❌ Faltan las variables de entorno (GOOGLE_API_KEY).")
        st.stop()
    
    st.divider()
    
    # Navegación estratégica entre Nuevo Análisis y el Historial
    st.header("Navegación")
    modo = st.radio("Elige una acción:", ["Analizar Nuevo CSV", "Ver Historial de Análisis"])
    
    st.divider()
    st.write("**Datasets de prueba:** [Kaggle Beginner Datasets](https://www.kaggle.com/datasets/ahmettezcantekin/beginner-datasets)")

# =======================
# LÓGICA PRINCIPAL
# =======================

if modo == "Analizar Nuevo CSV":
    st.write("Sube un CSV para obtener un análisis rápido de su estructura y recomendaciones de modelado.")
    
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if uploaded_file is not None:
        nombre_archivo = uploaded_file.name
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Vista previa de los datos")
        st.dataframe(df.head())
        
        dimensiones = f"{df.shape[0]} filas x {df.shape[1]} columnas"
        st.write(f"**Dimensiones:** {dimensiones}")

        if st.button("Generar Análisis de ML"):
            columnas = df.columns.tolist()
            info_tipos = df.dtypes.astype(str).to_dict()
            muestra = df.head(3).to_markdown() 
            
            prompt = f"""
            Eres un Lead Data Scientist presentando una evaluación técnica de un dataset a tu equipo.
            Analiza el siguiente dataset basándote en esta muestra:
            - Columnas: {columnas}
            - Tipos de datos: {info_tipos}
            - Muestra de datos:
            {muestra}

            Genera un reporte estructurado y analítico, manteniendo la facilidad de lectura mediante viñetas. Utiliza EXCLUSIVAMENTE el siguiente formato exacto:

            🎯 **Naturaleza del Problema:**
            - [Describe de qué trata el dataset y define claramente el objetivo predictivo].

            ⚠️ **Alertas de Calidad y Preprocesamiento:**
            - 🚩 **[Nombre de la variable o problema]:** [Justifica por qué es un riesgo y propón cómo abordarlo (ej. técnica de imputación, manejo de desbalanceo, codificación)].
            - 🚩 **[Nombre de la variable o problema]:** [Analiza aspectos como distribuciones sesgadas, alta cardinalidad o multicolinealidad, y la solución propuesta].

            🧠 **Estrategia de Modelado (Baseline vs. Avanzado):**
            - **Modelo de Línea Base:** [Sugiere un modelo simple e interpretable] para establecer un punto de referencia de rendimiento inicial.
            - **Modelo Principal:** [Recomienda ensambles basados en árboles como XGBoost o Random Forest].
            - **Justificación Técnica:** [Explica detalladamente el "por qué". Justifica cómo el modelo principal maneja mejor las no linealidades, la dimensionalidad o las interacciones complejas de este dataset específico frente al baseline].

            📏 **Auditoría Técnica y Evaluación:**
            - **Métricas:** [Sugiere la métrica principal (ej. F1-score, AUC-ROC) y explica por qué se ajusta a la penalización de errores de este caso particular].
            - **Explicabilidad (XAI):** [Detalla cómo integrar herramientas como SHAP o LIME para realizar una auditoría técnica del modelo, explicando qué valor aportará entender las predicciones a nivel local y global en este contexto].

            REGLA: Mantén un tono riguroso y argumentativo. Explica siempre el razonamiento detrás de tus elecciones en un máximo de dos o tres oraciones por viñeta. No uses introducciones ni conclusiones genéricas.
            """
            
            with st.spinner("Analizando la estructura de los datos con IA..."):
                try:
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                    texto_respuesta = response.text
                    
                    st.markdown("---")
                    st.markdown(texto_respuesta)
                    
                    # Guardar en MongoDB
                    registro = {
                        "archivo": nombre_archivo,
                        "dimensiones": dimensiones,
                        "fecha": datetime.now(),
                        "analisis": texto_respuesta
                    }
                    coleccion_historial.insert_one(registro)
                    st.toast("✅ Análisis guardado en el historial de MongoDB", icon="💾")
                    
                except Exception as e:
                    st.error(f"Error durante el análisis: {e}")

elif modo == "Ver Historial de Análisis":
    st.header("🗄️ Historial de Archivos Analizados")
    
    # Recuperar documentos ordenados por fecha (más recientes primero)
    historial_docs = list(coleccion_historial.find().sort("fecha", -1))
    
    if not historial_docs:
        st.info("Aún no hay análisis guardados en la base de datos.")
    else:
        # Crear un diccionario para el selectbox para mostrar el nombre y la fecha
        opciones_historial = {
            str(doc["_id"]): f"📄 {doc['archivo']} - {doc['fecha'].strftime('%Y-%m-%d %H:%M')}" 
            for doc in historial_docs
        }
        
        seleccion_id = st.selectbox(
            "Selecciona un análisis previo:", 
            options=list(opciones_historial.keys()), 
            format_func=lambda x: opciones_historial[x]
        )
        
        # Encontrar el documento seleccionado
        doc_seleccionado = next(doc for doc in historial_docs if str(doc["_id"]) == seleccion_id)
        
        st.markdown("---")
        st.subheader(f"Resumen de: {doc_seleccionado['archivo']}")
        st.write(f"**Dimensiones registradas:** {doc_seleccionado.get('dimensiones', 'N/A')}")
        
        with st.container():
            st.markdown(doc_seleccionado["analisis"])