import os
import cv2
import imutils
import time
import argparse

from feature_extraction.eye_blink_extractor import EyeBlinkExtractor
from feature_extraction.facial_changes_extractor import FacialChangesExtractor
from feature_extraction.outsite_changes_extractor import OutsiteChangesExtractor

from utils.writer_csv import WriterCSV
from utils.dlib_face_detector import DlibFaceDetector


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument("-r", "--realPath",
                    help="Path to the folder where the real videos are stored.",
                    type=str,
                    default=None)
parser.add_argument("-f", "--fakePath",
                    help="Path to the folder where the fake videos are stored.",
                    type=str,
                    default=None)
parser.add_argument("-o", "--outCsv",
                    help="Output CSV features file.",
                    type=str,
                    default="features.csv")

args = parser.parse_args()


VIDEO_RESIZED_WIDTH = 400


extractors = [EyeBlinkExtractor("eye_bink"),
              FacialChangesExtractor("facial_changes"),
              OutsiteChangesExtractor("outsite_changes")]

writer = WriterCSV(args.outCsv)
face_detector = DlibFaceDetector()


def process_videos(dir_path: str, is_deepfake: bool):
    filenames = os.listdir(dir_path)
    total_files = len(filenames)
    for idx, filename in enumerate(filenames):
        print("Processing {}: {}/{} - {}".format("fake" if is_deepfake else "real", idx + 1, total_files, filename))
        start_process_time = time.time()

        vidcap = cv2.VideoCapture("{}/{}".format(dir_path, filename))

        if not vidcap.isOpened():
            print("Cannot open video '{}'. Continuing...".format(filename))
            vidcap.release()
            continue

        detection_counter = 0
        total_frames_counter = 0
        while True:
            ret, frame = vidcap.read()
            if not ret:
                break

            total_frames_counter += 1

            frame = imutils.resize(frame, width=VIDEO_RESIZED_WIDTH)

            if face_detector.detect(frame):
                for extractor in extractors:
                    extractor.process(face_detector)
            else:
                for extractor in extractors:
                    extractor.reset_prev_frame()
                continue

            detection_counter += 1

        if detection_counter > 1:
            # saving features for a video
            all_features = { }
            base_features = {"filename": "\"{}\"".format(filename[:-4]),
                             "is_deepfake": int(is_deepfake),
                             "face_detection_ratio": round(detection_counter / total_frames_counter, 4)}

            all_features = {**all_features, **base_features}
            for extractor in extractors:
                extractor_features = extractor.result()
                all_features = {**all_features, **extractor_features}

            writer.write(all_features)

            print("face_detection_ratio: {}".format(all_features["face_detection_ratio"]))
        else:
            print("Any detection on video '{}'".format(filename))

        print("extraction in: {}s".format(round(time.time() - start_process_time, 1)))

        vidcap.release()
        for extractor in extractors:
            extractor.reset()


if __name__ == "__main__":
    if args.realPath is not None:
        process_videos(args.realPath, is_deepfake=False)
    else:
        print("No real videos provided!")

    if args.fakePath is not None:
        process_videos(args.fakePath, is_deepfake=True)
    else:
        print("No fake videos provided!")

    print("Feature extraction finished!")
