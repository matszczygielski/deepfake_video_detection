import numpy as np
import cv2
from scipy.signal import convolve2d
import diplib

import mahotas as mt

from feature_extraction.abstract_feature_extractor import AbstractFeatureExtractor
from utils.metric_analizer import MetricAnalizer
from utils.dlib_face_detector import DlibFaceDetector


class OutsiteChangesExtractor(AbstractFeatureExtractor):
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
            "O-H": MetricAnalizer("O-H"),
            "O-S": MetricAnalizer("O-S"),
            "O-V": MetricAnalizer("O-V"),
            "O-Contast": MetricAnalizer("O-Contast"),
            "O-Noise": MetricAnalizer("O-Noise"),
            "O-Sharp": MetricAnalizer("O-Sharp"),
            "O-SRM3-R": MetricAnalizer("O-SRM3-R"),
            "O-SRM3-G": MetricAnalizer("O-SRM3-G"),
            "O-SRM3-B": MetricAnalizer("O-SRM3-B"),
            "O-SRM5-R": MetricAnalizer("O-SRM5-R"),
            "O-SRM5-G": MetricAnalizer("O-SRM5-G"),
            "O-SRM5-B": MetricAnalizer("O-SRM5-B")
        }

        self.face_outsite_padding_width_pct = 15
        self.face_outsite_padding_height_pct = 50
        self.face_padding_width_pct = 0
        self.face_padding_height_pct = 0


    def process(self, detector: DlibFaceDetector) -> None:
        image_outsite = detector.get_image_outsite_on_facial_bb(self.face_outsite_padding_width_pct, self.face_outsite_padding_height_pct,
                                                                   self.face_padding_width_pct, self.face_padding_height_pct)

        mask_outsite = detector.get_mask_outsite_on_facial_bb(self.face_outsite_padding_width_pct, self.face_outsite_padding_height_pct,
                                                                 self.face_padding_width_pct, self.face_padding_height_pct)

        mask_outsite3 = np.zeros_like(image_outsite)
        mask_outsite3[:,:,0] = mask_outsite
        mask_outsite3[:,:,1] = mask_outsite
        mask_outsite3[:,:,2] = mask_outsite

        # h s v
        outsite_rgb = cv2.bitwise_and(image_outsite, mask_outsite3)
        outsite_hsv = cv2.cvtColor(outsite_rgb, cv2.COLOR_BGR2HSV)
        outsite_area = (mask_outsite.shape[0] * mask_outsite.shape[1]) - detector.get_facial_bb_area(self.face_padding_width_pct,
                                                                                                     self.face_padding_height_pct)
        outsite_sum_hsv = cv2.sumElems(outsite_hsv)
        self.metrics["O-H"].update(outsite_sum_hsv[0] / outsite_area)
        self.metrics["O-S"].update(outsite_sum_hsv[1] / outsite_area)
        self.metrics["O-V"].update(outsite_sum_hsv[2] / outsite_area)

        # contrast
        outsite_grey = cv2.cvtColor(outsite_rgb, cv2.COLOR_BGR2GRAY)
        outsite_locs = np.where(mask_outsite == 255)
        outsite_pixels = outsite_grey[outsite_locs[0], outsite_locs[1]]
        outsite_min = float(np.min(outsite_pixels))
        outsite_max = float(np.max(outsite_pixels))
        outsite_contrast = 0
        if (outsite_min + outsite_max) > 0:
            outsite_contrast = (outsite_max - outsite_min) / (outsite_max + outsite_min)
        self.metrics["O-Contast"].update(outsite_contrast)

        # noise
        sigma = diplib.EstimateNoiseVariance(outsite_grey)
        self.metrics["O-Noise"].update(sigma)

        # sharpness
        lap = cv2.Laplacian(outsite_grey, cv2.CV_32F)
        _, std = cv2.meanStdDev(lap)
        self.metrics["O-Sharp"].update(std[0][0]**2)

        # SRM filter
        out_srm = np.absolute(cv2.filter2D(outsite_rgb, -1, self.mask_srm_3))
        out_srm_sum = cv2.sumElems(out_srm)
        self.metrics["O-SRM3-R"].update(out_srm_sum[0] / outsite_area)
        self.metrics["O-SRM3-G"].update(out_srm_sum[1] / outsite_area)
        self.metrics["O-SRM3-B"].update(out_srm_sum[2] / outsite_area)

        out_srm = np.absolute(cv2.filter2D(outsite_rgb, -1, self.mask_srm_5))
        out_srm_sum = cv2.sumElems(out_srm)
        self.metrics["O-SRM5-R"].update(out_srm_sum[0] / outsite_area)
        self.metrics["O-SRM5-G"].update(out_srm_sum[1] / outsite_area)
        self.metrics["O-SRM5-B"].update(out_srm_sum[2] / outsite_area)


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
