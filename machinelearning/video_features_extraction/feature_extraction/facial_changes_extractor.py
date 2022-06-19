import numpy as np
import cv2
import math
import diplib

from skimage.feature import hog

from feature_extraction.abstract_feature_extractor import AbstractFeatureExtractor
from utils.metric_analizer import MetricAnalizer
from utils.dlib_face_detector import DlibFaceDetector


class FacialChangesExtractor(AbstractFeatureExtractor):
    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.mask_srm_5 = np.array([[-1,  2, -2,  2, -1],
                                    [ 2, -6,  8, -6,  2],
                                    [-2,  8, -12, 8, -2],
                                    [ 2, -6,  8, -6,  2],
                                    [-1,  2, -2,  2, -1]]) / 12

        self.mask_srm_3 = np.array([[-1, 2,  -1],
                                    [2, -4, 2],
                                    [-1, 2, -1]]) / 4

        self.metrics = {
            "F-H": MetricAnalizer("F-H"),
            "F-S": MetricAnalizer("F-S"),
            "F-V": MetricAnalizer("F-V"),
            "F-Contast": MetricAnalizer("F-Contast"),
            "F-Noise": MetricAnalizer("F-Noise"),
            "F-Sharp": MetricAnalizer("F-Sharp"),
            "F-Rot": MetricAnalizer("F-Rot"),
            "F-Pos": MetricAnalizer("F-Pos"),
            "F-SRM3-R": MetricAnalizer("F-SRM3-R"),
            "F-SRM3-G": MetricAnalizer("F-SRM3-G"),
            "F-SRM3-B": MetricAnalizer("F-SRM3-B"),
            "F-SRM5-R": MetricAnalizer("F-SRM5-R"),
            "F-SRM5-G": MetricAnalizer("F-SRM5-G"),
            "F-SRM5-B": MetricAnalizer("F-SRM5-B")
        }

        self.hog_orientations = 9
        self.hog_pixels_per_cell = (8, 8)

        for i in range(self.hog_orientations
                       * self.hog_pixels_per_cell[0]
                       * self.hog_pixels_per_cell[1]):
            self.metrics["F-hog-{}".format(i)] = MetricAnalizer("F-hog-{}".format(i))


    def process(self, detector: DlibFaceDetector) -> None:
        image_facial = detector.get_image_facial_on_facial_bb()
        mask_facial_hull = detector.get_mask_facial_hull_on_facial_bb()
    
        mask_facial_hull3 = np.zeros_like(image_facial)
        mask_facial_hull3[:,:,0] = mask_facial_hull
        mask_facial_hull3[:,:,1] = mask_facial_hull
        mask_facial_hull3[:,:,2] = mask_facial_hull

        # h s v
        facial_hull_rgb = cv2.bitwise_and(image_facial, mask_facial_hull3)

        facial_hull_hsv = cv2.cvtColor(facial_hull_rgb, cv2.COLOR_BGR2HSV)
        facial_area = detector.get_facial_area()
        facial_sum_hsv = cv2.sumElems(facial_hull_hsv)
        self.metrics["F-H"].update(facial_sum_hsv[0] / facial_area)
        self.metrics["F-S"].update(facial_sum_hsv[1] / facial_area)
        self.metrics["F-V"].update(facial_sum_hsv[2] / facial_area)

        # contrast
        facial_hull_grey = cv2.cvtColor(facial_hull_rgb, cv2.COLOR_BGR2GRAY)
        
        facial_locs = np.where(mask_facial_hull == 255)
        facial_pixels = facial_hull_grey[facial_locs[0], facial_locs[1]]
        facial_min = float(np.min(facial_pixels))
        facial_max = float(np.max(facial_pixels))
        facial_contrast = 0
        if facial_min + facial_max > 0:
            facial_contrast = (facial_max - facial_min) / (facial_max + facial_min)
        self.metrics["F-Contast"].update(facial_contrast)

        # noise
        facial_grey = cv2.cvtColor(detector.get_image_facial_on_facial_bb(), cv2.COLOR_BGR2GRAY)
        sigma = diplib.EstimateNoiseVariance(facial_grey)
        self.metrics["F-Noise"].update(sigma)

        # sharpness
        lap = cv2.Laplacian(facial_grey, cv2.CV_32F)
        _, std = cv2.meanStdDev(lap)
        self.metrics["F-Sharp"].update(std[0][0]**2)

        # facial position & rotation
        facial_hull = detector.get_facial_hull()
        rotated_rect = cv2.minAreaRect(facial_hull)
        (center, dims, angle) = rotated_rect
        box = cv2.boxPoints(rotated_rect)
        box = np.int0(box)
        if abs(box[0][1] - box[1][1]) > dims[1]/2:
            angle = -90 + angle
        angle *= -1
        self.metrics["F-Rot"].update(angle)
        self.metrics["F-Pos"].update(math.sqrt(center[0]**2 + center[1]**2))

        # SRM filter
        out_srm = np.absolute(cv2.filter2D(facial_hull_rgb, -1, self.mask_srm_3))
        out_srm_sum = cv2.sumElems(out_srm)
        self.metrics["F-SRM3-R"].update(out_srm_sum[0] / facial_area)
        self.metrics["F-SRM3-G"].update(out_srm_sum[1] / facial_area)
        self.metrics["F-SRM3-B"].update(out_srm_sum[2] / facial_area)

        out_srm = np.absolute(cv2.filter2D(facial_hull_rgb, -1, self.mask_srm_5))
        out_srm_sum = cv2.sumElems(out_srm)
        self.metrics["F-SRM5-R"].update(out_srm_sum[0] / facial_area)
        self.metrics["F-SRM5-G"].update(out_srm_sum[1] / facial_area)
        self.metrics["F-SRM5-B"].update(out_srm_sum[2] / facial_area)

        # HOG
        resized_img = cv2.resize(facial_hull_rgb, (64, 64))
        fd, hog_image = hog(resized_img,
                            orientations=self.hog_orientations,
                            pixels_per_cell=self.hog_pixels_per_cell, 
                            cells_per_block=(1, 1),
                            visualize=True, multichannel=True)

        for idx, fd_element in enumerate(fd):
            self.metrics["F-hog-{}".format(idx)].update(fd_element)


    def result(self) -> dict:
        result = { }
        for _, analizer in self.metrics.items():
            result = {**result, **analizer.get_metrics()}

        return result


    def reset_prev_frame(self):
        for key, _ in self.metrics.items():
            self.metrics[key].prev_value = None
            self.metrics[key].last_three_values = []


    def reset(self):
        for key, _ in self.metrics.items():
            self.metrics[key] = MetricAnalizer(key)
