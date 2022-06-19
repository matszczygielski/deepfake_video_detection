from utils.dlib_face_detector import DlibFaceDetector

class AbstractFeatureExtractor:
    def __init__(self, name: str) -> None:
        self.name = name


    def process(self, detector: DlibFaceDetector) -> None:
        raise NotImplementedError("Extractor {}: Function 'process' not implemented".format(self.name))


    def result(self) -> dict:
        raise NotImplementedError("Extractor {}: Function 'result' not implemented".format(self.name))


    def reset(self):
        raise NotImplementedError("Extractor {}: Function 'reset' not implemented".format(self.name))


    def reset_prev_frame(self):
        raise NotImplementedError("Extractor {}: Function 'reset_prev_frame' not implemented".format(self.name))
