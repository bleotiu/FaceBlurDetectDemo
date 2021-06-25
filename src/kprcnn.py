from PIL import Image
from torchvision.transforms import transforms
from math import sqrt

import numpy as np

import torch
import torchvision
import cv2
import argparse


def get_euclidean_dist(xa, ya, xb, yb):
    return sqrt((xb - xa) ** 2 + (yb - ya) ** 2)


def infer_on_image(image_path, aspect_ratio=1.33, human_detection_threshold=1e-3):
    faces_bboxes_list = []
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=False,
                        help='path to the input data')
    args = vars(parser.parse_args())

    args['input'] = image_path
    # transform to convert the image to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # initialize the model
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                                   num_keypoints=17)
    # set the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the modle on to the computation device and set to eval mode
    model.to(device).eval()

    image_path = args['input']
    image = Image.open(image_path).convert('RGB')
    # NumPy copy of the image for OpenCV functions
    orig_numpy = np.array(image, dtype=np.float32)
    # convert the NumPy image to OpenCV BGR format
    orig_numpy = cv2.cvtColor(orig_numpy, cv2.COLOR_RGB2BGR) / 255.
    # transform the image
    image = transform(image)
    # add a batch dimension
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)

    output_image = orig_numpy

    for output in outputs:
        all_det_scores = output['scores'].cpu().detach().numpy()  # human detection scores
        all_keypoints = output['keypoints'].cpu().detach().numpy()  # 17 human keypoints
        for i, (keypoint_coordinates_list, human_detection_score) in enumerate(
                list(zip(all_keypoints, all_det_scores))):
            if human_detection_score > human_detection_threshold:  # between 0 and 1
                nose_x, nose_y, nose_proba = keypoint_coordinates_list[0]
                left_ear_x, left_ear_y, left_ear_proba = keypoint_coordinates_list[3]
                right_ear_x, right_ear_y, right_ear_proba = keypoint_coordinates_list[4]
                if nose_proba and left_ear_proba and right_ear_proba:
                    dist_between_ears = get_euclidean_dist(left_ear_x, left_ear_y, right_ear_x, right_ear_y)
                    xmin, ymin = int(nose_x - dist_between_ears // aspect_ratio), int(
                        nose_y - dist_between_ears // aspect_ratio)
                    xmax, ymax = int(nose_x + dist_between_ears // aspect_ratio), int(
                        nose_y + dist_between_ears // aspect_ratio)

                    start_point = (xmin, ymin)
                    end_point = (xmax, ymax)
                    # Blue color in BGR
                    color = (255, 0, 0)

                    # Line thickness of 2 px
                    thickness = 2

                    # cv2.rectangle(output_image, start_point, end_point, color, thickness)
                    # faces_bboxes_list.append([xmin, ymin, xmax, ymax])
                    faces_bboxes_list.append([xmin, ymin, xmax - xmin, ymax - ymin])

    return faces_bboxes_list
