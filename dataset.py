import pandas as pd
import pydicom
import numpy as np
import re

class DataPreparer:

    def __init__(self):
        self.X = None
        self.y = None

    def open_registeredImages(self):

        allImages = []
        for i in range(1, 100):
            cas = []
            for j in range(52):
                dcm_file = ".../CasesRegistrated/Cas{}/Image{}.dcm".format(i, j)
                dcm_data = pydicom.read_file(dcm_file)
                image = dcm_data.pixel_array
                cas.append(image)
            allImages.append(cas)        
        return np.asarray(allImages)


    def open_annotations(self):
        
        file = ".../Annotations/Liste_Angio_3D_Anom.xlsx"
        labels = pd.read_excel(file)
        labels = labels[['Valsava Sinuses', 'Sino-tubular junction',
                         'Ascending aorta', 'Horizontal aorta', 'Isthmus',
                         'Descending aorta']].copy().drop([0]).iloc[0:100]
        return labels


    def get_data(self, landmark):
        
        images = self.open_registeredImages()
        labels = self.open_annotations()
        indexMissing = labels[str(landmark)].index[labels[str(landmark)].apply(np.isnan)]
        self.X = np.delete(images, indexMissing, 0)
        self.y = labels[[str(landmark)]].dropna().values
        return self.X, self.y