from face_detection import mtcnn_detector
import matplotlib
from matplotlib import pyplot
from PIL import Image
import os

# Création du dossier cible
dir_cible = r'C://Users//admin//0_DOSSIER_PYTHON//fondamentaux//python//DeepLearning//KERAS//CNN//Reconnaissance Faciale//face_reco_maison//data//brut//portraits'
dir_sortie = 'C://Users//admin//0_DOSSIER_PYTHON//fondamentaux//python//DeepLearning//KERAS//CNN//Reconnaissance Faciale//face_reco_maison//data//ok//'

# fonction d'enregistrement des visages détectés sous forme d'array en image (ici des jpg)
def save_face(arrayface, name):
        matplotlib.image.imsave(os.path.join(dir_sortie+name),arrayface) # attend en param (path ou nom, et nom de l'array)
    
# Instanciation du mtcnn
monMTCNN = mtcnn_detector.MTCnnDetector
numero = int(00)

# Boucle de traitement des images
for images in os.listdir(dir_cible):
    if images.endswith(".jpg") or images.endswith(".png"):        
       for faces in monMTCNN(os.path.join(dir_cible, images)).process_image(): # renvoie un tableau contenant les np_arrays des visages détectés
            numero = numero + 1
            save_face(faces,str(numero)+images)
    else:
        continue


