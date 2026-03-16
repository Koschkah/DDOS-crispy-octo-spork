import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="DDoS Analízis Dashboard", layout="wide")


st.markdown("""
    <style>
    .plot-card {
        border-radius: 10px;
        background-color: #f9f9f9;
        padding: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    iframe { border: none !important; }
    </style>
    """, unsafe_allow_html=True)


st.sidebar.title("Vezérlőpult")
selected_set = st.sidebar.selectbox("Adathalmaz", ["SetA", "SetB", "SetC", "SetD", "Combined"])


def render_plot(file_key, title, height=450):
    path = f"vis/{selected_set}_{file_key}.html"
    if os.path.exists(path):
        with st.container():
            st.markdown(f"**{title}**")
            with open(path, 'r', encoding='utf-8') as f:
                html_data = f.read()
                components.html(html_data, height=height)
    else:
        st.warning(f"Hiányzó ábra: {title}")

st.title(f"Hálózati Forgalom Analízis - {selected_set}")
st.divider()




col_left, col_right = st.columns([1, 2])
with col_left:
    render_plot("class_balance", "Forgalom Típusok Aránya", height=400)
with col_right:
    
    render_plot("duration", "Események időtartama", height=400)


st.subheader("Forgalmi Intenzitás Vizsgálata")
render_plot("traffic", "Packet Speed vs Data Speed", height=600)


st.divider()
col_bot1, col_bot2 = st.columns(2)
with col_bot1:
    render_plot("corr", "Jellemzők Korrelációs Mátrixa", height=500)
with col_bot2:
    render_plot("source", "Forrás IP Diverzitás", height=500)


st.divider()
render_plot("codes", "Attack Code-ok eloszlása a kategóriák szerint", height=500)