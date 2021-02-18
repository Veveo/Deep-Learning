import streamlit as st
from fastai.vision.all import load_learner, Path, torch, PILImage
from PIL import Image
import pathlib
import json

# Fix pour le probleme de Posixpath
temp= pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath

# Importation des descriptions
f = open('C:/Users/admin/0_DOSSIER_PYTHON/FastAI/StreamLitChampi/description.json')
descript = json.load(f)

# Chargement du model
model = load_learner(Path("C:/Users/admin/0_DOSSIER_PYTHON/FastAI/StreamLitChampi/modelmushroom2.pkl"))
# Hack pour ajouter un css custom
st.markdown('<style>' + open('C:/Users/admin/0_DOSSIER_PYTHON/FastAI/StreamLitChampi/style.css').read() + '</style>', unsafe_allow_html=True)


st.title('Mushroom Detector')
st.header("Identifiez vos champignons")

up=st.sidebar.file_uploader("Upload")
texte_ux="Envoyer une image de champignon en utilisant le menu de gauche"

if up != None:
    st.image(up)
    texte_ux="Bravo ! Lancons l'identification"
    
    if st.button("Identification",key="pred"):
        entry = PILImage.create(up)
        resultat=model.predict(entry)
        st.header(f"Votre champignon est de la famille des {resultat[0]}")
        description = descript[resultat[0]]
        texte_ux = description
        
else:
    image = Image.open('C:/Users/admin/0_DOSSIER_PYTHON/FastAI/StreamLitChampi/champi.jpg')
    st.image(image,use_column_width=True)       

st.write(texte_ux)

