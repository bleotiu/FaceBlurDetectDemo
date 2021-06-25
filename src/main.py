import cv2 as cv

from FaceBlurDetect import *


def demo():
    paths = [
        './data/fotbal.jpg',
        './data/tenis.jpg',
        './data/parada.jpg'
    ]

    # detectie cu mtcnn si blurare cu filtru uniform
    blur_detect = FaceBlurDetect()

    img = cv.imread(paths[0])
    cv.imshow('original', img)
    cv.waitKey(0)
    blurred_img = blur_detect.blur_image(img)
    cv.imshow('inainte de cenzura', img)
    cv.imshow('dupa cenzura', blurred_img)
    cv.waitKey(0)

    # detectie cu Haar si blurare cu filtru gaussian
    blur_detect = FaceBlurDetect(model_name='Haar', blurring_method='Gaussian')

    img = cv.imread(paths[1])
    blurred_img = blur_detect.blur_image(img)
    cv.imshow('inainte de cenzura', img)
    cv.imshow('dupa cenzura', blurred_img)
    cv.waitKey(0)

    # vom extrage doar locatiile detectiilor in format (x, y, w, h)
    blur_detect = FaceBlurDetect()
    detections = blur_detect.detect_faces(paths[2])
    print('Detectiile pentru ' + paths[2] + ' in format(x, y, w, h) sunt: ')
    print(detections)

    # vom cenzura o imagine folosind niste detectii aleatoare
    # furnizate exterior
    blur_detect = FaceBlurDetect(blurring_method='Sticker', sticker_color='green')
    random_detections = [
        [0, 0, 30, 25], # dreptunghi de 30x25 care incepe in coltul (0, 0)
        [57, 23, 40, 40] # patrat de latura 40 care incepe in coltul (57, 23)
    ]
    random_blurred_img = blur_detect.blur_image_portions(paths[1], random_detections)
    blurred_img = blur_detect.blur_image_portions(paths[2], detections)
    cv.imshow('imagine cenzurata aleator', random_blurred_img)
    cv.imshow('imagine cenzurata cu detectii precalculate', blurred_img)
    cv.waitKey(0)


if __name__ == '__main__':
    demo()