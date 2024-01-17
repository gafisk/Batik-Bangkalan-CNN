import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import folium
from streamlit_folium import folium_static
from assets import *

st.set_page_config(
    page_title="Sentimen Batik Bangkalan",
    page_icon="computer",
    layout="centered",
    initial_sidebar_state="expanded",
)

if 'df' not in st.session_state:
    st.session_state.df = None
if 'datas' not in st.session_state:
    st.session_state.datas = None
if 'data_tf' not in st.session_state:
    st.session_state.data_tf = None

with st.sidebar:
    selected = option_menu( 
        menu_title="Main Menu",
        options=["Home", "Dataset", "Preprocessing", "TF-IDF", "Evaluasi"],
        icons=["house", "book", "radioactive", "bezier2", "ubuntu"],
        menu_icon="menu-up",
        default_index=0,
    )

if selected == "Home":
    st.title("Performa CNN + TF-IDF")
    st.subheader("Dataset Batik Bangkalan")
    m = folium.Map(location=[-7.029867973140592, 112.74860070772233], zoom_start=13)
    tooltip = "Bangkalan"
    folium.Marker(
        [-7.029867973140592, 112.74860070772233], popup="Bangkalan", tooltip=tooltip
    ).add_to(m)
    folium_static(m)

if selected == "Dataset":
    st.title("Dataset Sentimen Batik Bangkalan")
    data = dataset()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Positif")
        st.subheader(data[1][0])
    with col2:
        st.subheader("Negatif")
        st.subheader(data[1][2])
    with col3:
        st.subheader("Netral")
        st.subheader(data[1][1])
    st.session_state.df = data[0]
    st.dataframe(data[0])

if selected == "Preprocessing":
    if st.session_state.df is not None:
        st.title("Preprocessing Data")
        data = st.session_state.df
        st.subheader("Data Sebelum Di Proses")
        st.dataframe(data)
        st.subheader("Data Hasil Preprocessing")
        st.session_state.datas = pd.read_excel("Preprocessing.xlsx")
        st.dataframe(st.session_state.datas)
    else:
        st.title("Warning!!!")
        st.subheader("Buka Menu Dataset Terlebih Dahulu")

if selected == "TF-IDF":
    if st.session_state.datas is not None:
        st.title("Preprocessing")
        st.subheader("Data Preprocessing")
        st.dataframe(st.session_state.datas)
        if st.button("Run TF-IDF"):
            data = tf_idf(st.session_state.datas)
            st.subheader("Hasil TF-IDF")
            st.dataframe(data[0])
            st.session_state.data_tf = data[1]
    else:
        st.title("Warning!!!")
        st.subheader("Buka Menu Preprocessing Terlebih Dahulu")

if selected == "Evaluasi":
    if st.session_state.data_tf is not None:
        st.title("Hasil Evaluasi")
        n = st.slider('Pilih Nilai K', min_value=2, max_value=20, value=5, step=1)
        hasil = fold(st.session_state.data_tf, st.session_state.datas, n)
        st.subheader("Rata - Rata Nilai")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.subheader("Loss")
            st.subheader(f"{round(hasil[1],1)} %")
        with col2:
            st.subheader("Akurasi")
            st.subheader(f"{round(hasil[2],1)} %")
        with col3:
            st.subheader("Presisi")
            st.subheader(f"{round(hasil[3],1)} %")
        with col4:
            st.subheader("Recall")
            st.subheader(f"{round(hasil[4],1)} %")
        with col5:
            st.subheader("F1 Score")
            st.subheader(f"{round(hasil[5],1)} %")
        st.subheader("Hasil Evaluasi")
        h1, h2 = st.columns(2)
        with h1:
            st.write(hasil[0])
        with h2:
            st.subheader("Lama Proses Evaluasi")
            st.subheader(hasil[6])
    else:
        st.title("Warning!!!")
        st.subheader("Tekan Proses TF-IDF di Menu TF-IDF")
