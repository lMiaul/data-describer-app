import streamlit as st
import pandas as pd
import pymongo
from datetime import datetime
from google import genai

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
        # Extraer URI desde los secretos
        uri = st.secrets["MONGODB_URI"]
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
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("API Key e infraestructura cargadas correctamente.")
    except KeyError:
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
            Eres un Lead Data Scientist presentando un dataset a tu equipo técnico. Tienes solo 30 segundos para captar su atención.
            Analiza el siguiente dataset basándote en esta muestra:
            - Columnas: {columnas}
            - Tipos de datos: {info_tipos}
            - Muestra de datos:
            {muestra}

            Genera un reporte ultra-conciso y altamente escaneable. DEBES usar EXCLUSIVAMENTE el siguiente formato exacto (respeta los emojis y las negritas):

            🎯 **Veredicto Rápido:** [Una sola oración contundente describiendo la naturaleza y posible objetivo predictivo del dataset].

            ⚠️ **Alertas de Calidad (Data Prep):**
            - 🚩 [Problema 1 en máximo 10 palabras, ej. categóricas a codificar].
            - 🚩 [Problema 2 en máximo 10 palabras, ej. posible desbalanceo].

            🧠 **Estrategia de Modelado:**
            - **Modelo ideal:** [Recomienda ensambles robustos como XGBoost o Random Forest si el caso lo amerita].
            - **Justificación:** [Razón puramente técnica en una sola línea].

            📏 **Auditoría y Métricas:**
            - **Optimizar para:** [Métrica clave sugerida, ej. F1-score, AUC-ROC].
            - **Explicabilidad (XAI):** [Sugiere cómo auditar el modelo usando SHAP o LIME en una sola línea].

            REGLA ESTRICTA: Prohibido usar párrafos largos. Prohibido incluir saludos, introducciones o conclusiones genéricas. Ve directo al grano.
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