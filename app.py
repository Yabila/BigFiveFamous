%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Configuración de página
st.set_page_config(page_title="Big Five - Clasificador de Clusters", layout="centered")

# 1. Cargar el dataset, modelo y escalador
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/gerrymay/fcaData/refs/heads/master/oceanbig5.csv"
    df = pd.read_csv(url)
    # Mapear las categorías
    categorias_dict = {
        1: 'Actor/actriz', 2: 'Cantante', 3: 'Modelo', 4: 'Tv, series',
        5: 'Radio', 6: 'Tecnología', 7: 'Deportes', 8: 'Política', 9: 'Escritor'
    }
    df['nombre_categoria'] = df['categoria'].map(categorias_dict)
    return df

try:
    df = load_data()
    kmeans_loaded = joblib.load('kmeans_bigFive_model.joblib')
    scaler_loaded = joblib.load('scaler_bigFive.joblib')
    
    # Asignar clusters al dataset original de la misma manera que en el notebook
    features = ['op', 'co', 'ex', 'ag', 'ne']
    df_scaled = scaler_loaded.transform(df[features])
    df['cluster'] = kmeans_loaded.predict(df_scaled)
except Exception as e:
    st.error(f"Error al cargar los recursos: {e}. Asegúrate de tener los archivos '.joblib' en el mismo directorio.")
    st.stop()

st.title("Clasificador de Personalidad - Big Five Test")
st.write("Ingresa tus puntajes para predecir tu cluster y compararte con famosos similares.")

# Formulario para entrada de datos
with st.sidebar:
    st.header("Tus Puntajes del Test")
    op_input = st.slider("Apertura Mental (op)", 0.0, 100.0, 50.0, 0.1)
    co_input = st.slider("Orden / Conciencia (co)", 0.0, 100.0, 50.0, 0.1)
    ex_input = st.slider("Extroversión (ex)", 0.0, 100.0, 50.0, 0.1)
    ag_input = st.slider("Amabilidad / Empatía (ag)", 0.0, 100.0, 50.0, 0.1)
    ne_input = st.slider("Neuroticismo (ne)", 0.0, 100.0, 50.0, 0.1)

# Procesar si el usuario presiona el botón
if st.button("Calcular Cluster"):
    # Preparar el dataframe del usuario
    user_df = pd.DataFrame([{
        'op': op_input,
        'co': co_input,
        'ex': ex_input,
        'ag': ag_input,
        'ne': ne_input
    }])
    
    # Escalar y Predecir
    user_scaled = scaler_loaded.transform(user_df)
    user_cluster = kmeans_loaded.predict(user_scaled)[0]
    
    st.success(f"### ¡Has sido asignado al **Cluster {user_cluster}**!")
    
    # Famosos en este cluster
    famosos_cluster = df[df['cluster'] == user_cluster]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Top 5 Famosos de este Cluster")
        # Mostramos los primeros 5 famosos del dataset en este cluster
        st.dataframe(famosos_cluster[['usuario', 'nombre_categoria', 'wordcount']].head(5))
        
    with col2:
        st.write("#### Estadísticas del Cluster")
        st.write(f"Total de famosos en este cluster: {len(famosos_cluster)}")
        st.write("Frecuencia de profesiones:")
        st.write(famosos_cluster['nombre_categoria'].value_counts())

    # Gráfico de Radar
    st.write("### Comparación de Perfil de Personalidad (Radar Chart)")
    
    # 1. Valores promedio del cluster asignado
    cluster_avg = famosos_cluster[features].mean().values.tolist()
    cluster_avg += cluster_avg[:1]
    
    # 2. Valores ingresados por el usuario
    user_values = [op_input, co_input, ex_input, ag_input, ne_input]
    user_values_loop = user_values + [user_values[0]]
    
    num_vars = len(features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Dibujar etiquetas
    plt.xticks(angles[:-1], features, color='black', size=12)
    ax.set_rlabel_position(0)
    
    max_val = max(max(user_values), max(cluster_avg)) + 10
    plt.ylim(0, max_val)
    
    # Graficar Promedio del Cluster
    ax.plot(angles, cluster_avg, linewidth=2, linestyle='solid', color='#1f77b4', label=f'Promedio Cluster {user_cluster}')
    ax.fill(angles, cluster_avg, color='#1f77b4', alpha=0.25)
    
    # Graficar Resultados Personales
    ax.plot(angles, user_values_loop, linewidth=2, linestyle='dashed', color='#d62728', label='Mis Resultados')
    ax.fill(angles, user_values_loop, color='#d62728', alpha=0.15)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    st.pyplot(fig)
