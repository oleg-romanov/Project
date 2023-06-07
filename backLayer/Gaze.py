import os
import cv2
import dlib
import math
import torch
import json
import numpy as np
import queue
import threading

from torchvision import transforms

class Tracker:
    def __init__(
        self,
        gpu=0,
        isDebug=False
    ):
        self.isDebug = isDebug
        self.l_eye_img = np.zeros((64, 64, 3))
        self.r_eye_img = np.zeros((64, 64, 3))
        dlib.cuda.set_device(gpu)
        self.cap = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)
    
    def get_frame(self):
        frame = self.q.get()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:
            landmarks = self.predictor(gray, face)

            right_eye_start_point_X = landmarks.part(36).x
            right_eye_start_point_Y = landmarks.part(36).y

            right_eye_end_point_X = landmarks.part(39).x
            right_eye_end_point_Y = landmarks.part(39).y

            left_eye_start_point_X = landmarks.part(42).x
            left_eye_start_point_Y = landmarks.part(42).y

            left_eye_end_point_X = landmarks.part(45).x
            left_eye_end_point_Y = landmarks.part(45).y

            right_eye_center_X = int((right_eye_start_point_X + right_eye_end_point_X) / 2)
            right_eye_center_Y = int((right_eye_start_point_Y + right_eye_end_point_Y) / 2)

            left_eye_center_X = int((left_eye_start_point_X + left_eye_end_point_X) / 2)
            left_eye_center_Y = int((left_eye_start_point_Y + left_eye_end_point_Y) / 2)

            if self.isDebug:
                # Рисуем синий круг в центре правого глаза
                cv2.circle(frame, (right_eye_center_X, right_eye_center_Y), 2, (255, 0, 0), 1)
                # Рисуем синий круг в центре левого глаза
                cv2.circle(frame, (left_eye_center_X, left_eye_center_Y), 2, (255, 0, 0), 1)

            # Обрезаем картинку, сначала указывается промежуток по оси Y, 
            # затем промежуток по оси X.
            r_eye_img = frame[
                right_eye_center_Y - (right_eye_center_X - right_eye_start_point_X):right_eye_center_Y + (right_eye_center_X - right_eye_start_point_X), 
                right_eye_start_point_X:right_eye_end_point_X
            ]
            l_eye_img = frame[
                left_eye_center_Y - (left_eye_center_X - left_eye_start_point_X):left_eye_center_Y + (left_eye_center_X - left_eye_start_point_X),
                left_eye_start_point_X:left_eye_end_point_X
            ]
            self.l_eye_img = cv2.resize(l_eye_img, (64, 64))
            self.r_eye_img = cv2.resize(r_eye_img, (64, 64))

            # cv2.imwrite("rightEye.png", right_eye_image)
            # cv2.imwrite("leftEye.png", left_eye_image)
            # cv2.imshow("Frame", frame)
            return (self.r_eye_img, self.r_eye_img)

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

class Predictor:
    def __init__(self, model, model_data, config_file=None):
        super().__init__()

        _, ext = os.path.splitext(model_data)
        if ext == ".ckpt":
            self.model = model.load_from_checkpoint(model_data)
        else:
            with open(config_file) as json_file:
                config = json.load(json_file)
            self.model = model(config)
            self.model.load_state_dict(torch.load(model_data))

        self.model.double()
        self.model.eval()

    # Предсказать
    # При определении функции можно использовать * , чтобы собрать переменное 
    # количество позиционных аргументов, переданных в функцию. 
    # Они помещаются в кортеж, поэтому указан *img_list, сюда из файла 
    # collect_data.py передаются left_eye и right_eye (l_eye, r_eye), имеющие тип Mat
    def predict(self, *img_list, head_angle=None):
        images = []
        for img in img_list:
            if not img.dtype == np.uint8:
                img = img.astype(np.uint8)
            img = transforms.ToTensor()(img).unsqueeze(0)
            img = img.double()
            images.append(img)

        with torch.no_grad():
            coords = self.model(*images)
            coords = coords.cpu().numpy()[0]

        return coords[0], coords[1]

if __name__ == "__main__":
    tracker = Tracker()

    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
        tracker.get_frame()

    tracker.close()