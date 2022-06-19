import numpy as np
import dlib
import cv2

from imutils import face_utils


class DlibFaceDetector:
    def __init__(self, face_detector_model: str = "hogsvm", num_of_unsample: int = 0) -> None:
        self.face_detector_model = face_detector_model

        if face_detector_model == "hogsvm":
            self.detector = dlib.get_frontal_face_detector()
        elif face_detector_model == "cnn_resnet":
            self.detector = dlib.cnn_face_detection_model_v1("utils/models/dlib_face_recognition_resnet_model_v1.dat")
        elif face_detector_model == "cnn_mmod":
            self.detector = dlib.cnn_face_detection_model_v1("utils/models/mmod_human_face_detector.dat")
        else:
            raise NotImplementedError("{} detector not implemented".format(face_detector_model))
        
        self.unsample = num_of_unsample

        self.predictor = dlib.shape_predictor("utils/models/shape_predictor_68_face_landmarks.dat")

        self.facial_landmarks = {
            "jawStart": face_utils.FACIAL_LANDMARKS_IDXS["jaw"][0],
            "jawEnd": face_utils.FACIAL_LANDMARKS_IDXS["jaw"][1],
            "rbStart": face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"][0],
            "rbEnd": face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"][1],
            "lbStart": face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"][0],
            "lbEnd": face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"][1],
            "leStart": face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0],
            "leEnd": face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1],
            "reStart": face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0],
            "reEnd": face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]
        }

        self.face_bb = None
        self.facial_hull = None
        self.facial_bb = None
        self.image_bgr = None
        self.left_eye_points = None
        self.right_eye_points = None
 

    def detect(self, image_bgr) -> bool:
        self.image_bgr = image_bgr
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector(image_gray, self.unsample)

        # No face detected
        if len(faces) == 0:
            self.face_bb = None
            self.facial_hull = None
            self.facial_bb = None
            self.image_bgr = None
            self.left_eye_points = None
            self.right_eye_points = None
            return False

        # More than one face detected
        if len(faces) > 1:
            self.face_bb = None
            self.facial_hull = None
            self.facial_bb = None
            self.image_bgr = None
            self.left_eye_points = None
            self.right_eye_points = None
            return False

        if self.face_detector_model == "hogsvm":
            face = faces[0]
        else:
            face = faces[0].rect

        self.face_bb = self.convert_and_trim_bb(self.image_bgr, face)

        shape = self.predictor(image_rgb, face)
        shape = face_utils.shape_to_np(shape)

        facial_points = np.vstack((shape[self.facial_landmarks["jawStart"]:self.facial_landmarks["jawEnd"]],
                                   shape[self.facial_landmarks["lbStart"]:self.facial_landmarks["lbEnd"]],
                                   shape[self.facial_landmarks["rbStart"]:self.facial_landmarks["rbEnd"]]))

        self.facial_hull = cv2.convexHull(facial_points)
        self.facial_bb = cv2.boundingRect(self.facial_hull)
        self.facial_bb = (max(0, self.facial_bb[0]), max(0, self.facial_bb[1]), self.facial_bb[2], self.facial_bb[3])

        self.left_eye_points = shape[self.facial_landmarks["leStart"]:self.facial_landmarks["leEnd"]]
        self.right_eye_points = shape[self.facial_landmarks["reStart"]:self.facial_landmarks["reEnd"]]

        self.left_eye_hull = cv2.convexHull(self.left_eye_points)
        self.right_eye_hull = cv2.convexHull(self.right_eye_points)
        self.left_eyebrow_hull = cv2.convexHull(shape[self.facial_landmarks["lbStart"]:self.facial_landmarks["lbEnd"]])
        self.right_eyebrow_hull = cv2.convexHull(shape[self.facial_landmarks["rbStart"]:self.facial_landmarks["rbEnd"]])

        return True


    def get_left_eye_points(self):
        return self.left_eye_points


    def get_right_eye_points(self):
        return self.right_eye_points


    def get_facial_hull(self):
        return self.facial_hull


    def get_face_bb(self, padding_w_pct: int = 0, padding_h_pct: int = 0):
        if padding_w_pct > 0 or padding_h_pct > 0:
            return self.extend_bb(self.image_bgr, self.face_bb, padding_w_pct, padding_h_pct)
        return self.face_bb


    def get_facial_bb(self, padding_w_pct: int = 0, padding_h_pct: int = 0):
        if padding_w_pct > 0 or padding_h_pct > 0:
            return self.extend_bb(self.image_bgr, self.facial_bb, padding_w_pct, padding_h_pct)
        return self.facial_bb


    def get_image(self):
        return self.image_bgr


    def get_image_face(self, padding_w_pct: int = 0, padding_h_pct: int = 0):
        if self.face_bb is None:
            return None

        if padding_w_pct > 0 or padding_h_pct > 0:
            (x, y, w, h) = self.extend_bb(self.image_bgr, self.face_bb, padding_w_pct, padding_h_pct)
        else:
            (x, y, w, h) = self.face_bb

        return self.image_bgr[y:y+h, x:x+w]
    

    def get_image_facial_on_face_bb(self, padding_w_pct: int = 0, padding_h_pct: int = 0):
        if self.face_bb is None:
            return None

        if padding_w_pct > 0 or padding_h_pct > 0:
            (x, y, w, h) = self.extend_bb(self.image_bgr, self.face_bb, padding_w_pct, padding_h_pct)
        else:
            (x, y, w, h) = self.face_bb

        return self.image_bgr[y:y+h, x:x+w]


    def get_image_facial_on_facial_bb(self, padding_w_pct: int = 0, padding_h_pct: int = 0):
        if self.facial_bb is None:
            return None

        if padding_w_pct > 0 or padding_h_pct > 0:
            (x, y, w, h) = self.extend_bb(self.image_bgr, self.facial_bb, padding_w_pct, padding_h_pct)
        else:
            (x, y, w, h) = self.facial_bb

        return self.image_bgr[y:y+h, x:x+w]


    def get_image_outsite_on_face_bb(self, padding_outsite_w_pct: int, padding_outsite_h_pct: int,
                                     padding_face_w_pct: int = 0, padding_face_h_pct: int = 0):

        if self.face_bb is None:
            return None

        extended_face_bb = self.extend_bb(self.image_bgr, self.face_bb, padding_face_w_pct, padding_face_h_pct)
        outsite_bb = self.extend_bb(self.image_bgr, extended_face_bb, padding_outsite_w_pct, padding_outsite_h_pct)

        (x, y, w, h) = outsite_bb

        return self.image_bgr[y:y+h, x:x+w]
    

    def get_image_outsite_on_facial_bb(self, padding_outsite_w_pct: int, padding_outsite_h_pct: int,
                                       padding_face_w_pct: int = 0, padding_face_h_pct: int = 0):

        if self.facial_bb is None:
            return None

        extended_facial_bb = self.extend_bb(self.image_bgr, self.facial_bb, padding_face_w_pct, padding_face_h_pct)
        outsite_bb = self.extend_bb(self.image_bgr, extended_facial_bb, padding_outsite_w_pct, padding_outsite_h_pct)

        (x, y, w, h) = outsite_bb

        return self.image_bgr[y:y+h, x:x+w]


    def get_mask_face_on_orginal(self, padding_w_pct: int = 0, padding_h_pct: int = 0):
        if self.face_bb is None:
            return None

        if padding_w_pct > 0 or padding_h_pct > 0:
            (x, y, w, h) = self.extend_bb(self.image_bgr, self.face_bb, padding_w_pct, padding_h_pct)
        else:
            (x, y, w, h) = self.face_bb

        mask = np.zeros((self.image_bgr.shape[0], self.image_bgr.shape[1]), dtype='uint8')
        mask = cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

        return mask


    def get_mask_facial_bb_on_orginal(self, padding_w_pct: int = 0, padding_h_pct: int = 0):
        if self.facial_bb is None:
            return None

        if padding_w_pct > 0 or padding_h_pct > 0:
            (x, y, w, h) = self.extend_bb(self.image_bgr, self.facial_bb, padding_w_pct, padding_h_pct)
        else:
            (x, y, w, h) = self.facial_bb

        mask = np.zeros((self.image_bgr.shape[0], self.image_bgr.shape[1]), dtype='uint8')
        mask = cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

        return mask


    def get_mask_facial_hull_on_orginal(self):
        if self.facial_hull is None:
            return None

        mask = np.zeros((self.image_bgr.shape[0], self.image_bgr.shape[1]), dtype='uint8')
        mask = cv2.drawContours(mask, [self.facial_hull], -1, 255, -1)

        return mask


    def get_mask_facial_hull_on_face(self, padding_w_pct: int = 0, padding_h_pct: int = 0):
        mask_facial = self.get_mask_facial_hull_on_orginal()

        if mask_facial is None or self.face_bb is None:
            return None

        if padding_w_pct > 0 or padding_h_pct > 0:
            (x, y, w, h) = self.extend_bb(self.image_bgr, self.face_bb, padding_w_pct, padding_h_pct)
        else:
            (x, y, w, h) = self.face_bb

        return mask_facial[y:y+h, x:x+w]


    def get_mask_facial_hull_on_facial_bb(self, padding_w_pct: int = 0, padding_h_pct: int = 0):
        mask_facial = self.get_mask_facial_hull_on_orginal()

        if mask_facial is None or self.facial_bb is None:
            return None

        if padding_w_pct > 0 or padding_h_pct > 0:
            (x, y, w, h) = self.extend_bb(self.image_bgr, self.facial_bb, padding_w_pct, padding_h_pct)
        else:
            (x, y, w, h) = self.facial_bb

        return mask_facial[y:y+h, x:x+w]


    def get_mask_outsite_on_face_bb(self, padding_outsite_w_pct: int, padding_outsite_h_pct: int,
                                    padding_face_w_pct: int = 0, padding_face_h_pct: int = 0):

        mask = self.get_mask_face_on_orginal(padding_face_w_pct, padding_face_h_pct)

        extended_face_bb = self.extend_bb(self.image_bgr, self.face_bb, padding_face_w_pct, padding_face_h_pct)
        outsite_bb = self.extend_bb(self.image_bgr, extended_face_bb, padding_outsite_w_pct, padding_outsite_h_pct)

        (x, y, w, h) = outsite_bb

        return cv2.bitwise_not(mask[y:y+h, x:x+w])


    def get_mask_outsite_on_facial_bb(self, padding_outsite_w_pct: int, padding_outsite_h_pct: int,
                                      padding_face_w_pct: int = 0, padding_face_h_pct: int = 0):

        mask = self.get_mask_facial_bb_on_orginal(padding_face_w_pct, padding_face_h_pct)

        extended_facial_bb = self.extend_bb(self.image_bgr, self.facial_bb, padding_face_w_pct, padding_face_h_pct)
        outsite_bb = self.extend_bb(self.image_bgr, extended_facial_bb, padding_outsite_w_pct, padding_outsite_h_pct)

        (x, y, w, h) = outsite_bb

        return cv2.bitwise_not(mask[y:y+h, x:x+w])


    def get_facial_area(self) -> float:
        if self.facial_hull is None:
            return 0
        return cv2.contourArea(self.facial_hull)


    def get_face_bb_area(self, padding_w_pct: int = 0, padding_h_pct: int = 0) -> float:
        if self.face_bb is None:
            return 0

        if padding_w_pct > 0 or padding_h_pct > 0:
            (x, y, w, h) = self.extend_bb(self.image_bgr, self.face_bb, padding_w_pct, padding_h_pct)
        else:
            (x, y, w, h) = self.face_bb

        return w * h


    def get_facial_bb_area(self, padding_w_pct: int = 0, padding_h_pct: int = 0) -> float:
        if self.facial_bb is None:
            return 0

        if padding_w_pct > 0 or padding_h_pct > 0:
            (x, y, w, h) = self.extend_bb(self.image_bgr, self.facial_bb, padding_w_pct, padding_h_pct)
        else:
            (x, y, w, h) = self.facial_bb

        return w * h


    def extend_bb(self, image, bb, padding_w_pct, padding_h_pct):
        (x, y, w, h) = bb
        padding_w_px = int(w * padding_w_pct/100)
        padding_h_px = int(h * padding_h_pct/100)
        x = max(0, x - padding_w_px)
        y = max(0, y - padding_h_px)
        w = min(image.shape[1] , w + 2*padding_w_px)
        h = min(image.shape[0] , h + 2*padding_h_px)

        return (x, y, w, h)

        
    def convert_and_trim_bb(self, image, rect):
        startX = rect.left()
        startY = rect.top()
        endX = rect.right()
        endY = rect.bottom()

        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(endX, image.shape[1])
        endY = min(endY, image.shape[0])

        w = endX - startX
        h = endY - startY

        return (startX, startY, w, h)
