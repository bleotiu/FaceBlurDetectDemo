from mtcnn.mtcnn import MTCNN
from kprcnn import *

import cv2 as cv
import numpy as np

import os.path
import copy


MODEL_NAMES = [
    'MTCNN',
    'Haar',
    'KPRCNN'
]
BLURRING_METHODS = [
    '2D_filter',
    'Gaussian',
    'Sticker'
]
STICKER_COLORS = [
    'black',
    'red',
    'yellow',
    'blue',
    'green',
    'orange',
    'purple',
    'white',
]
STICKER_COLORS_DICT = {
    'black': [0, 0, 0],
    'red': [0, 0, 255],
    'yellow': [0, 255, 255],
    'blue': [255, 0, 0],
    'green': [0, 255, 0],
    'orange': [0, 128, 255],
    'purple': [128, 0, 128],
    'white': [255, 255, 255]
}

HAAR_CLASSIFIER = 'haarcascade_frontalface_alt.xml'
HAAR_SCALE_FACTOR = 1.05

KPRCNN_HUMAN_THRESHOLD = 0.001
KPRCNN_ASPECT_RATIO = 1.33

BLURRING_KERNEL_SIZE = 21
BLURRING_KERNEL_MATRIX = np.ones((BLURRING_KERNEL_SIZE, BLURRING_KERNEL_SIZE), np.float32) / (BLURRING_KERNEL_SIZE * BLURRING_KERNEL_SIZE)


def _get_image_from_source(src_image):
    if isinstance(src_image, str):
        # Daca sursa este calea catre imagine
        # vom incarca intai imaginea
        return cv.imread(src_image)
    else:
        return src_image


def _check_detections(detections, subject='Detections array'):
    np_array = np.array(detections)
    if len(np_array.shape) != 2 or np_array.shape[1] != 4:
        raise ValueError(f'{subject} should have shape (N, 4), but it has shape {np_array.shape}')


def _check_model_name(model_name):
    if model_name not in MODEL_NAMES:
        raise ValueError(f'model_name should be from {MODEL_NAMES}')


def _check_sticker_color(sticker_color):
    if sticker_color not in STICKER_COLORS:
        raise ValueError(f'sticker_color should be from {STICKER_COLORS}')


class FaceBlurDetect:
    """
    Clasa folosita pentru a cenzura si detecta fete intr-o poza, un set de poze sau un videoclip
    Pentru partea de detectie avem urmatoarele optiuni:
        1) Sa folosim modele de detectie ale clasei (pentru care vom modifica parametrul model_name):
            1.1) Pentru model_name = 'MTCNN' vom folosi modelul MTCNN (https://pypi.org/project/mtcnn/)
            1.2) Pentru model_name = 'Haar' vom folosi modelul Haar Cascades
            (https://docs.opencv.org/master/d2/d99/tutorial_js_face_detection.html)
        2) Sa folosim modele antrenate separat (pentru care vom modifica parametrul model).
        In acest caz modelul furnizat clasei trebuie sa primeasca drept parametru o imagine si sa
        returneze o lista de detectii de forma [x, y, w, h] unde:
            (x, y) - cel mai de jos si din stanga punct al dreptunghiului in care se incadreaza fata
            w - latimea dreptunghiului in care se incadreaza fata
            h - inaltimea dreptunghiului in care se incadreaza fata
    Pentru partea de cenzurare avem urmatoarele optiuni(pentru care vom modifica blurring_method):
        1) pentru blurring_method = '2D_filter' vom folosi un filtru de 21x21 cu care fiecare pixel al
        unei imagini va deveni media aritmetica a tuturor vecinilor
        2) pentru blurring_method = 'Gaussian' vom folosi un filtru gaussian de 43x43 care spre deosebire
        de filtrul 2D mentionat anterior va converti valoarea fiecarui pixel din imagine folosind o medie
        ponderata a vecinilor, vecinii mai apropiati avand o valoare mai mare
        3) pentru blurring_method = 'Sticker' pentru fiecare detectie vom "lipi" un sticker de culoarea
        sticker_color pe imaginea furnizata

    """
    def __init__(
            self,
            model_name='MTCNN',
            model=None,
            blurring_method='2D_filter',
            sticker_color='black'):
        """

        :param model_name:
        :param model:
        :param blurring_method:
        :param sticker_color:
        """

        # assert model is not None or model_name in MODEL_NAMES, f'The given model should not be None' \
        #                                                        f'or the model_name should be from {MODEL_NAMES}'
        #
        # assert blurring_method in BLURRING_METHODS, f'The given blurring_method should be from {BLURRING_METHODS}'
        #
        # if blurring_method == 'Sticker':
        #     assert sticker_color in STICKER_COLORS, f'The given sticker_color should be from {STICKER_COLORS}'

        self._model_name = copy.deepcopy(model_name)
        self._model = copy.deepcopy(model)
        self.blurring_method = copy.deepcopy(blurring_method)
        self.sticker_color = copy.deepcopy(sticker_color)
        self._detection_model = None
        self._is_instantiated = False
        # for model_name in MODEL_NAMES:
        #     self._is_instantiated[model_name] = False

    def _find_faces(self, img, image_path=''):
        # assert self._model is not None or self._model_name in MODEL_NAMES,
        # f'The given model should not be None or the model_name should be from {MODEL_NAMES}'
        if self._model is not None:
            # aplicam modelul furnizat de utilizator pe imagine
            detections = self._model(img)

            # verificam ca rezultatul modelului va avea forma buna
            _check_detections(detections, subject='Model output')

            return detections

        else:
            # Vom folosi un model preimplementat
            _check_model_name(self._model_name)

            if self._model_name == 'MTCNN':
                if not self._is_instantiated:
                    self._detection_model = MTCNN()
                    self._is_instantiated = True
                detections = self._detection_model.detect_faces(img)

                bboxes = [detection['box'] for detection in detections]
                return bboxes

            elif self._model_name == 'Haar':
                if not self._is_instantiated:
                    self._detection_model = cv.CascadeClassifier(cv.data.haarcascades + HAAR_CLASSIFIER)
                    self._is_instantiated = True

                detections = self._detection_model.detectMultiScale(img, scaleFactor=HAAR_SCALE_FACTOR)
                return detections

            elif self._model_name == 'KPRCNN':
                detections = infer_on_image(image_path,
                                            aspect_ratio=KPRCNN_ASPECT_RATIO,
                                            human_detection_threshold=KPRCNN_HUMAN_THRESHOLD)

                return detections

    def _blur_full_image(self, img):
        if self.blurring_method == 'Sticker':
            _check_sticker_color(self.sticker_color)
            blurred_image = np.zeros(img.shape, np.uint8)
            blurred_image[:, :] = STICKER_COLORS_DICT[self.sticker_color]
            return blurred_image

        elif self.blurring_method == '2D_filter':
            blurred_image = cv.filter2D(img, -1, BLURRING_KERNEL_MATRIX)
            return blurred_image

        elif self.blurring_method == 'Gaussian':
            blurred_image = cv.GaussianBlur(img, (2 * BLURRING_KERNEL_SIZE + 1, 2 * BLURRING_KERNEL_SIZE + 1), 0)
            return blurred_image

    def _blur_image_portions(self, img, detections):
        _check_detections(detections)

        # Vom construi o masca de dimensiunea pozei pe care vrem sa o cenzuram
        # si vom marca toate zonele pe care vrem sa le cenzuram cu True
        blurring_mask = np.zeros(img.shape, np.bool)

        for detection in detections:
            x_min = detection[0]
            y_min = detection[1]
            x_max = detection[0] + detection[2]
            y_max = detection[1] + detection[3]

            # Reversed dimension because of how opencv saves a given image
            blurring_mask[y_min: y_max, x_min: x_max] = True

        # Ne vom folosi de imaginea cenzurata complet pentru a obtine
        # imaginea cenzurata doar in zonele precizate
        fully_blurred_image = self._blur_full_image(img)

        # Construim noua imagine folosind imaginea initiala, pe cea cenzurata complet
        # si masca de mai sus
        partially_blurred_image = np.where(blurring_mask, fully_blurred_image, img)

        return partially_blurred_image

    def blur_image_portions(self, src_image, detections):
        current_image = _get_image_from_source(src_image)

        return self._blur_image_portions(current_image, detections)

    def blur_image(self, src_image):
        current_image = _get_image_from_source(src_image)

        # Gasim detectiile pentru imaginea data
        detections = self._find_faces(current_image, src_image)

        # Cenzuram portiunile detectate
        blurred_image = self._blur_image_portions(current_image, detections)

        return blurred_image

    def detect_faces(self, src_image):
        current_image = _get_image_from_source(src_image)

        detections = self._find_faces(current_image, src_image)

        return detections

    def blur_image_array(self, src_images):
        blurred_images = []
        for src_image in src_images:
            blurred_images.append(self.blur_image(src_image))

        return blurred_images

    def blur_image_directory(self, src_directory):
        blurred_images = []
        # Vom scana folder-ul furnizat si vom cenzura toate imaginile gasite
        for subdir, dirs, files in os.walk(src_directory):
            for image_name in files:
                current_image = cv.imread(os.path.join(subdir, image_name))
                # Daca fisierul curent nu este o imagine current_image va fii None
                if current_image is not None:
                    # Daca imaginea a fost citita cu succes o vom cenzura si adauga
                    # la rezultat
                    blurred_images.append(self.blur_image(current_image))

        return blurred_images

    def _blurred_frames_from_video(self, capture):
        blurred_frames = []

        while capture.isOpened():
            ret, frame = capture.read()
            if ret:
                blurred_frames.append(self.blur_image(frame))
            else:
                break
        capture.release()
        return blurred_frames

    def blurred_frames_from_video(self, src_video):
        to_release = False
        if isinstance(src_video, str):
            capture = cv.VideoCapture(src_video)
            to_release = True
        else:
            capture = src_video

        blurred_frames = self._blurred_frames_from_video(capture)

        if to_release:
            capture.release()

        return blurred_frames

    def make_video_from_frames(self, frames, video_path,
                               frames_per_second=20.0,
                               frame_size=(640, 480)):
        fourcc = cv.VideoWriter_fourcc(*'DIVX')
        out = cv.VideoWriter(video_path, fourcc, frames_per_second, frame_size)

        for frame in frames:
            out.write(frame)

        out.release()

    def blur_video(self, src_video_path, dst_video_path):
        capture = cv.VideoCapture(src_video_path)

        blurred_frames = self._blurred_frames_from_video(capture)

        fps = capture.get(cv.CAP_PROP_FPS)
        width, height = capture.get(cv.CAP_PROP_FRAME_WIDTH), capture.get(cv.CAP_PROP_FRAME_HEIGHT)

        self.make_video_from_frames(blurred_frames, dst_video_path,
                                    frames_per_second=fps, frame_size=(width, height))

        capture.release()


