from PIL import Image
from matplotlib import pyplot
from mtcnn import MTCNN
from numpy import asarray
from skimage import io
import os


class MTCnnDetector:

    def __init__(self, image_path):
        self.detector = MTCNN()
        self.image = io.imread(image_path)

    def process_image(self, plot=False): # fonction principale
        faces = self.__detect_face();
        resized_face_list = [] # creation d'une liste comme une image peut contenir plusieurs visages
        for f in faces:
            extracted_face = self.__extract_face(f)
            resized_face = self.__resize_img_to_face(extracted_face)
            resized_face_list.append(resized_face)
            if plot:
                self.__plot_face(resized_face)
        return resized_face_list

    def __detect_face(self): # détection des visages
        return self.detector.detect_faces(self.image)

    def __extract_face(self, face): # extraction de la position des des visages à partir des box
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        return self.image[y1:y2, x1:x2]

    def __resize_img_to_face(self, face): 
        image = Image.fromarray(face)
        image = image.resize((128,128))
        return asarray(image)

    def __plot_face(self, face): # fonction d'affichage des visagec détectés
        pyplot.imshow(face)
        pyplot.show()
        pyplot.close()

    #def plot_save(self,face,name):
        #pyplot.savefig(os.path.join('C://Users//admin//0_DOSSIER_PYTHON//fondamentaux//python//DeepLearning//KERAS//CNN//Reconnaissance Faciale//face_reco_maison//data//ok//',name))
    