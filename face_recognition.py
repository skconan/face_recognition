# 3rd party dependencies
import matplotlib.pyplot as plt
import numpy as np
import cv2

# project dependencies
from deepface import DeepFace
from deepface.modules import verification
from deepface.commons.logger import Logger

logger = Logger()


class FaceRecognition:
    def __init__(
        self,
        model_name="ArcFace",
        detector_backend="retinaface",
        # distance_metric="euclidean",
        distance_metric="cosine",
    ):
        self.model = DeepFace.build_model(
            task="facial_recognition", model_name=model_name
        )
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.detector_backend = detector_backend
        self.target_size = self.model.input_shape
        self.verify_threshold = verification.find_threshold(
            model_name=self.model_name, distance_metric=distance_metric
        )

    def detect(self, img: np.ndarray):
        face_info = DeepFace.extract_faces(
            img_path=img, detector_backend=self.detector_backend
        )
        if face_info is None:
            logger.error("No face detected!")
            return False, None

        face_info = face_info[0]
        area = face_info["facial_area"]
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]
        face_norm = face_info["face"]
        face_crop = img[y : y + h, x : x + w]
        face_norm = cv2.resize(face_norm, self.target_size)
        face_crop = cv2.resize(face_crop, self.target_size)
        return True, face_norm, face_crop

    def get_verify_threshold(self):
        return self.verify_threshold

    def cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def get_embedding_vector(self, img: np.ndarray):
        embedding_objs = DeepFace.represent(
            img,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            align=True,
        )
        return embedding_objs[0]["embedding"]

    def verify(self, img1: np.ndarray, img2: np.ndarray):
        results = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=self.model_name,
            distance_metric=self.distance_metric,
            detector_backend=self.detector_backend,
        )
        print(results)
        return results["verified"], round(results["distance"], 4)
