import cv2 as cv
import os
import numpy as np
import ntpath
import glob
from parameters import *


def show_detections_without_ground_truth(detections, scores, file_names, dir_test, params: Parameters):
    test_images_path = os.path.join(dir_test, '*.jpg')
    test_files = glob.glob(test_images_path)

    for test_file in test_files:
        image = cv.imread(test_file)
        short_file_name = ntpath.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]

        for idx, detection in enumerate(current_detections):
            cv.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
            cv.putText(image, 'score:' + str(current_scores[idx])[:4], (detection[0], detection[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # cv.imwrite(os.path.join(params.dir_save_files, "detections_" + short_file_name), image)
        # print('Apasa orice tasta pentru a continua...')
        cv.imshow('image', np.uint8(image))
        cv.waitKey(0)


def show_detections_with_ground_truth(detections, scores, file_names, params: Parameters):
    ground_truth_bboxes = np.loadtxt(params.path_annotations, dtype='str')
    test_images_path = os.path.join(params.dir_test_examples, '*.jpg')
    test_files = glob.glob(test_images_path)

    for test_file in test_files:
        image = cv.imread(test_file)
        short_file_name = ntpath.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]

        for idx, detection in enumerate(current_detections):
            cv.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
            cv.putText(image, 'score:' + str(current_scores[idx])[:4], (detection[0], detection[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        annotations = ground_truth_bboxes[ground_truth_bboxes[:, 0] == short_file_name]

        # show ground truth bboxes
        for detection in annotations:
            cv.rectangle(image, (int(detection[1]), int(detection[2])), (int(detection[3]), int(detection[4])), (0, 255, 0), thickness=1)

        # cv.imwrite(os.path.join(params.dir_save_files, "detections_" + short_file_name), image)
        # print('Apasa orice tasta pentru a continua...')
        cv.imshow('image', np.uint8(image))
        cv.waitKey(0)
