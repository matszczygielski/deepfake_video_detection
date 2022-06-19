# source
# https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/

from scipy.spatial import distance as dist

from feature_extraction.abstract_feature_extractor import AbstractFeatureExtractor
from utils.dlib_face_detector import DlibFaceDetector


class EyeBlinkExtractor(AbstractFeatureExtractor):
    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.eye_ar_thresh = 0.23
        self.eye_ar_consec_frames = 3

        self.blink_frames_counter = 0
        self.total_frames_counter = 0
        self.total_blinks = 0

    def process(self, detector: DlibFaceDetector) -> None:
        leftEAR = self.eye_aspect_ratio(detector.get_left_eye_points())
        rightEAR = self.eye_aspect_ratio(detector.get_right_eye_points())
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < self.eye_ar_thresh:
            self.blink_frames_counter += 1
        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if self.blink_frames_counter >= self.eye_ar_consec_frames:
                self.total_blinks += 1
            # reset the eye frame counter
            self.blink_frames_counter = 0

        self.total_frames_counter += 1


    def result(self) -> dict:
        eye_blink_ratio = 0
        if self.total_frames_counter > 0:
            eye_blink_ratio = self.total_blinks / self.total_frames_counter

        return {
            "eye-blink-ratio": round(eye_blink_ratio, 4)
        }


    def reset(self):
        self.blink_frames_counter = 0
        self.total_frames_counter = 0
        self.total_blinks = 0


    def reset_prev_frame(self):
        pass


    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
	    # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        # return the eye aspect ratio
        return ear
