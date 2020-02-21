import numpy as np
import cv2

def rotate_images(self, images, angle):
    
        imagesRotated = []
        for i in range(images.shape[0]):
            image = images[i]
            (h, w) = image.shape[:2]
            medio = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(medio, angle, 1.0)
            imagen_rotada = cv2.warpAffine(image, M, (w, h))
            imagesRotated.append(imagen_rotada)
        return np.asarray(imagesRotated)


def dataAugmentation(images, labels):

        all_images, all_labels = [], []
        mean, sigma = 0, 0.001
        for i in range(images.shape[0]):
            image, label = images[i], labels[i]
            all_images.append(image)
            all_labels.append(label)
            for i in range(30):
                #Gaussian Noise
                gaussian = np.random.normal(mean, sigma, (16, 180, 180))
                all_images.append(image + gaussian)
                all_labels.append(label)
                #Image Rotation
                angle = np.random.uniform(-7, 7, 1)
                rotatedImage = rotate_images(image, int(angle))
                all_images.append(rotatedImage)
                all_labels.append(label)
        return np.asarray(all_images), np.asarray(all_labels)

