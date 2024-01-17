import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import folium
from streamlit_folium import folium_static
import streamlit_antd_components as sac
from assets import *

st.set_page_config(
    page_title="Sentimen Batik Bangkalan",
    page_icon="computer",
    layout="centered",
    initial_sidebar_state="expanded",
)

if 'df_upload' not in st.session_state:
    st.session_state.df_upload = None



if 'df' not in st.session_state:
    st.session_state.df = None
if 'datas' not in st.session_state:
    st.session_state.datas = None
if 'data_tf' not in st.session_state:
    st.session_state.data_tf = None
if 'tfidf_df' not in st.session_state:
    st.session_state.tfidf_df = None
if 'vocab' not in st.session_state:
    st.session_state.vocab = None

with st.sidebar:
    selected = sac.menu([
        sac.MenuItem('Home', icon="house"),
        sac.MenuItem('New Data', icon='database-add',
                     children=[
                         sac.MenuItem('Upload Data', icon='cloud-upload'),
                         sac.MenuItem('Cooming Soon', icon='emoji-sunglasses-fill'),
                    ]),
        sac.MenuItem('Self Training', icon='android2',
                     children=[
                         sac.MenuItem('Dataset', icon='book'),
                         sac.MenuItem('Preprocessing', icon='radioactive'),
                         sac.MenuItem('TF-IDF', icon='bezier2'),
                         sac.MenuItem('Training', icon='yin-yang'),
                    ]),
    ], open_all=True)

if selected == "Home":
    st.title("Performa CNN + TF-IDF")
    st.subheader("Dataset Batik Bangkalan")
    m = folium.Map(location=[-7.029867973140592, 112.74860070772233], zoom_start=13)
    tooltip = "Bangkalan"
    folium.Marker(
        [-7.029867973140592, 112.74860070772233], popup="Bangkalan", tooltip=tooltip
    ).add_to(m)
    folium_static(m)

if selected == "Upload Data":
    st.title("Upload Your Dataset")
    st.subheader("Oke, Kerjakan besok, ngantuk")
    upload_file = st.file_uploader("Pilih file Excel", type=["xlsx"])
    if upload_file is not None:
        st.success("File berhasil diunggah.")
        st.subheader("Preview Data:")
        st.session_state.df_upload = pd.read_excel(upload_file)
        if st.session_state.df_upload is not None:
            st.write(st.session_state.df_upload)
    else:
        st.warning("Silakan unggah file excel untuk melihat data.")

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
            st.session_state.tfidf_df = data[0]
            st.session_state.data_tf = data[1]
            st.session_state.vocab = data[2]
    else:
        st.title("Warning!!!")
        st.subheader("Buka Menu Preprocessing Terlebih Dahulu")

if selected == "Training":
    st.title("Proses Training")
    if st.session_state.data_tf is not None:
        sac.divider(label='Parameter CNN', icon='ubuntu', align='center', color='gray')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Pilih Nilai Epoch")
            epoch = st.slider('', min_value=50, max_value=500, value=100, step=50)
        with col2:
            st.write("Pilih Nilai Batch")
            batch = st.slider('', min_value=32, max_value=320, value=64, step=32)
        with col3:
            st.write("Fungsi Aktivasi")
            scol1, scol2 = st.columns(2)
            with scol1:
                layer1 = st.selectbox('Layer 1', ('relu', 'sigmoid', 'softmax'))
            with scol2:
                layer2 = st.selectbox('Layer 2', ('relu', 'sigmoid', 'softmax'))
        sac.divider(label='K-Fold', icon='ubuntu', align='center', color='gray')
        k = st.slider('Pilih Nilai K', min_value=2, max_value=20, value=5, step=1)
        if st.button("Training"):
            sac.divider(label='Hasil Training', icon='ubuntu', align='center', color='gray')
            hasil = fold(st.session_state.tfidf_df, layer1, layer2, st.session_state.vocab, k, st.session_state.data_tf, st.session_state.datas, epoch, batch )
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
