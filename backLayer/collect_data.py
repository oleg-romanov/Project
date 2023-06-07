import os
import sys
import pygame
import cv2
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import socket
import struct
import math

from scipy.stats import beta
from pygame.locals import *
from collections import deque
from Gaze import Tracker
from Gaze import Predictor
from Models import EyesModel
from MySocket import MySocket

class Target:
    def __init__(self, position, speed, radius=10, color=(255, 255, 255)):
        super().__init__()
        self.x = position[0]
        self.y = position[1]
        self.speed = speed
        self.radius = radius
        self.color = color
        self.moving = False

    def render(self, screen):
        # Рисуем белый круг
        pygame.draw.circle(
            surface=screen, color=(255, 255, 255), center=(self.x, self.y), radius=self.radius + 1
        )
        pygame.draw.circle(surface=screen, color=self.color, center=(self.x, self.y), radius=self.radius)

    def move(self, target_location, ticks):
        distantion_per_tick = self.speed * ticks / 1000

        # Убеждаемся, что цель проходит фиксированное расстояние в каждом кадре
        if (
            abs(self.x - target_location[0]) <= distantion_per_tick
            and abs(self.y - target_location[1]) <= distantion_per_tick
        ):
            self.moving = False
            self.color = (255, 0, 0)
        else:
            self.moving = True
            self.color = (0, 255, 0)
            current_vector = pygame.Vector2(x=self.x, y=self.y)
            new_vector = pygame.Vector2(x= target_location[0], y= target_location[1])
            # Направление
            towards = (new_vector - current_vector).normalize()

            self.x += towards[0] * distantion_per_tick
            self.y += towards[1] * distantion_per_tick

data_dirs = (
    "data/left_eye",
    "data/right_eye",
)
# Если этих директорий еще нет, то создаем их
for d in data_dirs:
    if not os.path.exists(d):
        os.makedirs(d)

data_file_path = "data/positions.csv"
data_file_exists = os.path.isfile(data_file_path)
data_file = open(data_file_path, "a", newline="")
csv_writer = csv.writer(data_file, delimiter=",")

image_size = 64

if not data_file_exists:
    csv_writer.writerow(["id", "x", "y"])


def get_calibration_zones(width, height, target_radius):
    # Получаем координаты для 9 точек калибровки
    xs = (0 + target_radius, width // 2, width - target_radius)
    ys = (0 + target_radius, height // 2, height - target_radius)
    zones = []
    for x in xs:
        for y in ys:
            zones.append((x, y))
    # Перемешиваем список
    random.shuffle(zones)
    return zones

def get_undersampled_region(region_map, map_scale):
    min_coords = np.where(region_map == np.min(region_map))
    idx = random.randint(0, len(min_coords[0]) - 1)
    return (min_coords[0][idx] * map_scale, min_coords[1][idx] * map_scale)

# Setup pygame
image_size = 64
target_radius = 15
target_speed = 280

map_scale = 10
black = (0, 0, 0)
gray = (200, 200, 200)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
bg = black
bg_should_increase = True
clock = pygame.time.Clock()
ticks = 0
frame_count = 0
record_frame_rate = 30

# selection_screen = True
selection_screen = False
calibrate_screen = False
# collect_state = 0
# collect_screen = True
collect_screen = False
# track_screen = False
track_screen = True
calibrate_idx = 0
# only save every x frames
skip_frames = 3
# increase probability of screen edge targets
focus_edges = True
# *only* target screen edges
only_edges = False
# beta distribution params for edge sampling
beta_a = 0.4
beta_b = 0.4
# number of frames to avg across for tracked eye location
avg_window_length = 5

if track_screen:
    pygame.init()
    pygame.mouse.set_visible(0)
    font_normal = pygame.font.SysFont(None, 30)
    font_small = pygame.font.SysFont(None, 20)
    pygame.display.set_caption("Calibrate and Collect")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    width, height = pygame.display.get_surface().get_size()
    print(f"width: {width}, height: {height}")

    center = (width // 2, height // 2)
    webcam_surface = pygame.Surface(
        (image_size * 2, image_size * 2)
    )

try:
    region_map = np.load("data/region_map.npy")
except FileNotFoundError:
    region_map = np.zeros(
        (int(width / map_scale), int(height / map_scale))
    )

if not track_screen:
    collect_start_region = get_undersampled_region(region_map, map_scale)

    calibration_zones = get_calibration_zones(width, height, target_radius)

track_x = deque(
    [0] * avg_window_length, maxlen=avg_window_length
)
track_y = deque(
    [0] * avg_window_length, maxlen=avg_window_length
)
track_error = deque(
    [0] * (avg_window_length * 2), maxlen=avg_window_length * 2
)
if track_screen:
# Create Target
    target = Target(
        # position=center,
        position=(720, 900),
        speed=target_speed,
        # radius=target_radius
        radius=4
    )

# Create Detector
# detector = Tracker(output_size=image_size)
detector = Tracker()

eyesModel = EyesModel()

# Create Predictor
predictor = Predictor(
    EyesModel,
    model_data="trained_model/eyetracking_model.pt",
    config_file="trained_model/eyetracking_config.json"
)

def get_num_images():
    len_right_eyes = len(os.listdir("data/right_eye"))
    len_left_eyes = len(os.listdir("data/left_eye"))
    if len_left_eyes > len_right_eyes:
        return len_left_eyes
    else:
        return len_right_eyes

num_images = get_num_images()

def save_data(
    num_images,
    l_eye,
    r_eye,
    targetx,
    targety,
):
    data_id = num_images + 1

    for (path, img) in zip(data_dirs, (l_eye, r_eye)):
        cv2.imwrite("{}/{}.jpg".format(path, data_id), img)

    csv_writer.writerow([data_id, targetx, targety])

    region_map[
        int(targetx / map_scale), int(targety / map_scale)
    ] += 1

    return data_id

def cleanup():
    np.save("data/region_map.npy", region_map)
    data_file.close()
    detector.close()
    pygame.quit()
    sys.exit(0)

def clamp_value(x, max_value):
    # Restrict values to a range
    if x < 0:
        return 0
    if x > max_value:
        return max_value
    return x

def send_data(cleint_sock, x, y):
    data = f"{x},{y}".encode()
    client_sock.sendall(data)
    client_sock.close()

def calculate_distance(x0, y0, x1, y1):
    return math.sqrt((x1 - x0)**2 + (y1 - y0)**2)

if track_screen:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    sock.bind(('localhost', 9097))
    sock.listen(1)
    client_sock, client_address = sock.accept()
    print('Подключение от клиента', client_address)
flag = False
minimum = math.inf
while True:
    if not track_screen:
        screen.fill(bg)

    # Get current frame from the detector
    frame_count += 1
    try: 
        r_eye, l_eye = detector.get_frame()
    except TypeError:
        print("Оба глаза не распознаны, соединение прекращено")
        continue

    # Selection screen
    if selection_screen:
        text1 = font_normal.render(
            "(1) Calibrate | (2) Collect | (3) Track", True, white
        )
        text2 = font_normal.render(
            "(c) Toggle camera | (s) Show stats | (esc) Quit", True, white
        )
        screen.blit(text1, (10, h * 0.3))
        screen.blit(text2, (10, h * 0.6))

        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                w, h = pygame.display.get_surface().get_size()
                calibration_zones = get_calibration_zones(
                    w, h, target_radius
                )
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                cleanup()
            elif event.type == KEYDOWN and event.key == K_c:
                show_webcam = not show_webcam
            elif event.type == KEYDOWN and event.key == K_s:
                show_stats = not show_stats
            elif event.type == KEYDOWN and event.key == K_1:
                selection_screen = False
                calibrate_screen = True
                target.moving = False
                target.color = blue
            elif event.type == KEYDOWN and event.key == K_2:
                selection_screen = False
                collect_screen = True
                target.color = green
            elif event.type == KEYDOWN and event.key == K_3:
                selection_screen = False
                track_screen = True
                target.color = red


    # Calibration screen
    if calibrate_screen:
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                w, h = pygame.display.get_surface().get_size()
                calibration_zones = get_calibration_zones(
                    w, h, target_radius=target_radius
                )
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                cleanup()
            elif event.type == KEYDOWN and event.key == K_c:
                show_webcam = not show_webcam
            elif event.type == KEYDOWN and event.key == K_s:
                show_stats = not show_stats
            elif event.type == KEYDOWN and event.key == K_SPACE:
                if calibrate_idx < len(calibration_zones):
                    num_images = save_data(
                        num_images,
                        l_eye,
                        r_eye,
                        target.x,
                        target.y,
                    )
                calibrate_idx += 1

        if calibrate_idx < len(calibration_zones):
            target.x, target.y = calibration_zones[calibrate_idx]
            target.render(screen)
        elif calibrate_idx == len(calibration_zones):
            screen.fill(black)
            text = font_normal.render("Done", True, white)
            screen.blit(text, text.get_rect(center=screen.get_rect().center))
        elif calibrate_idx > len(calibration_zones):
            calibrate_idx = 0
            selection_screen = True
            calibrate_screen = False
    
    # Сильно временно чтобы скипнуть сбор данных и использовать собранные ранее:
    collect_state = 2
    # Data collection screen
    if collect_screen:
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                w, h = pygame.display.get_surface().get_size()
                calibration_zones = get_calibration_zones(
                    w, h, target_radius
                )
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                cleanup()
            elif event.type == KEYDOWN and event.key == K_c:
                show_webcam = not show_webcam
            elif event.type == KEYDOWN and event.key == K_s:
                show_stats = not show_stats
            elif event.type == KEYDOWN and event.key == K_SPACE:
                collect_state += 1
                target.moving = False

        if collect_state == 0:
            target.x = collect_start_region[0]
            target.y = collect_start_region[1]
            target.render(screen)
        elif collect_state == 1:
            if not target.moving:
                if only_edges:
                    new_x = random.choice([0, w])
                    new_y = random.choice([0, h])
                    center = (new_x, new_y)
                elif focus_edges:
                    new_x = (
                        beta.rvs(beta_a, beta_b, size=1) * w
                    )[0]
                    new_y = (
                        beta.rvs(beta_a, beta_b, size=1) * h
                    )[0]
                    center = (new_x, new_y)
                else:
                    center = get_undersampled_region(region_map, map_scale)

            if frame_count % skip_frames == 0:
                frame_count = 0
                num_images = save_data(
                    num_images,
                    l_eye,
                    r_eye,
                    target.x,
                    target.y,
                )
            target.move(center, ticks)
            target.render(screen)
        elif collect_state == 2:
            screen.fill(black)
            text = font_normal.render("Done", True, white)
            screen.blit(text, text.get_rect(center=screen.get_rect().center))
            # fit model
            eyesModel.fitModel()
            break

        elif collect_state > 2:
            collect_state = 0
            selection_screen = True
            collect_screen = False

    
    # Track screen
    if track_screen:
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                print()
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                cleanup()
            elif event.type == KEYDOWN and event.key == K_c:
                show_webcam = not show_webcam
            elif event.type == KEYDOWN and event.key == K_s:
                show_stats = not show_stats
            elif event.type == KEYDOWN and event.key == K_SPACE:
                selection_screen = True
                track_screen = False

        x_hat, y_hat = predictor.predict(
            l_eye,
            r_eye
        )
        track_x.append(x_hat)
        track_y.append(y_hat)

        weights = np.arange(1, avg_window_length + 1)
        weights_error = np.arange(1, (avg_window_length * 2) + 1)

        test_x = np.average(track_x, weights=weights)
        test_y = np.average(track_y, weights=weights)

        info_byte = struct.pack('b', 1)
        data = info_byte + struct.pack('ff', test_x, test_y)

        try:
            client_sock.sendall(data)
        except socket.error as e:
            print("Соединение разорвано клиентом")
            client_sock.close()
            sock.close()
            break

        data = f"{test_x},{test_y}".encode()
        client_sock.sendall(data)

        print(f'x: {test_x}, y: {test_y}')
        target.x = test_x
        target.y = test_y
        target.radius = np.average(track_error, weights=weights_error)

        target.render(screen)

        # if flag == False:
        #     # flag = True
        #     result = calculate_distance(target.x, target.y, test_target.x, test_target.y)
        #     if result < minimum:
        #         minimum = result
        #         print(f"minimum = {minimum}")
        #         print("----------------------------------------------")
        #         print(f"Координата целевой точки: x= {test_target.x}, y= {test_target.y}")
        #         print(f"Координата предсказанной точки: x= {target.x}, y= {target.y}")
        #         print("----------------------------------------------")

    ticks = clock.tick(record_frame_rate)

    if not track_screen: 
        pygame.display.update()

if track_screen:
    client_sock.close()
    sock.close()
