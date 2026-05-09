import streamlit as st
import pandas as pd
from google import genai

# =======================
# CONFIGURACIÓN
# =======================
st.set_page_config(page_title="ML Data Describer", page_icon="📊", layout="wide")
st.title("📊 Data Describer para Machine Learning")
st.write("Sube un CSV para obtener un análisis rápido de su estructura, posibles problemas y recomendaciones de modelado.")

# =======================
# BARRA LATERAL (SIDEBAR)
# =======================
with st.sidebar:
    st.header("🔑 Configuración")
    # Extraer la API Key de los secretos (o pedirla si prefieres fallback manual)
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("API Key cargada correctamente desde Secrets.")
    except KeyError:
        st.error("❌ Faltan las variables de entorno. Configura GOOGLE_API_KEY.")
        st.stop()
    
    st.divider()
    
    st.header("📂 Datasets de prueba recomendados")
    st.write("¿No tienes un archivo a la mano? Descarga uno de los datasets para principiantes desde el siguiente enlace de Kaggle:")
    st.markdown("[🔗 Kaggle: Beginner Datasets](https://www.kaggle.com/datasets/ahmettezcantekin/beginner-datasets)")
    
    st.write("**Sugerencias interesantes del repositorio:**")
    st.markdown("""
    - 🏦 **`bank.csv`**: Ideal para predecir si un cliente suscribirá un depósito (Clasificación).
    - 🏠 **`house.csv`**: Clásico para predecir precios de viviendas (Regresión).
    - 🩺 **`diabetes.csv`**: Bueno para evaluar métricas de salud y falsos negativos.
    - 🛒 **`amazon.csv`**: Para explorar cómo el modelo sugiere procesar texto o NLP.
    """)
    st.info("Descarga cualquier archivo CSV del enlace anterior y súbelo en el panel principal para probar el motor de IA.")

# =======================
# CARGA DE DATOS PRINCIPAL
# =======================
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Leer datos
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())
    
    st.write(f"**Dimensiones:** {df.shape[0]} filas x {df.shape[1]} columnas")

    # Botón para accionar la IA
    if st.button("Generar Análisis de ML"):
        
        # Preparar un resumen del dataset para enviarlo a Gemini
        columnas = df.columns.tolist()
        info_tipos = df.dtypes.astype(str).to_dict()
        muestra = df.head(3).to_markdown() # Formateo en Markdown usando tabulate
        
        prompt = f"""
        Eres un experto en Data Science. Analiza el siguiente dataset basándote en esta muestra:
        - Columnas: {columnas}
        - Tipos de datos: {info_tipos}
        - Muestra de datos:
        {muestra}

        Proporciona un análisis estructurado que incluya:
        1. **Resumen:** ¿De qué parece tratar este dataset?
        2. **Calidad de Datos:** Identifica posibles problemas (ej. necesidad de limpieza, variables categóricas a codificar, sospecha de clases desbalanceadas).
        3. **Sugerencias de Modelado:** Recomienda enfoques predictivos. Si aplica, sugiere modelos potentes como ensambles (XGBoost, Random Forest) y justifica por qué.
        4. **Métricas y Auditoría:** Sugiere qué métricas usar (ej. F1-score, AUC-ROC) e indica si herramientas de explicabilidad (XAI) como SHAP o LIME serían valiosas para este caso de uso.
        """
        
        with st.spinner("Analizando la estructura de los datos con IA..."):
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                st.markdown("---")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Error al conectar con la API: {e}")