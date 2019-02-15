
# Dataset visualization tool
# Original By: Comma.ai and Chris Gundling
# Revised and used by Neil Nie


import matplotlib.backends.backend_agg as agg
from i3d import Inception3D
from os import listdir
import numpy as np
import cv2
import pygame
import pylab
import pandas as pd
import configs


pygame.init()
size = (640, 660)
pygame.display.set_caption("self driving data viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
screen.set_alpha(None)

camera_surface = pygame.surface.Surface((640, 480),0,24).convert()
clock = pygame.time.Clock()

PI_RAD = (180 / np.pi)
red = (255, 0, 0)
blue = (0, 0, 255)

# ***** get perspective transform for images *****
from skimage import transform as tf

rsrc = \
    [[30.45456230828867, 120.00743250075844],
     [109.3055617352614, 80.46865203761757],
     [114.86050156739812, 70.83953551083698],
     [129.74572757609468, 60.48459567870026],
     [132.98164627363735, 50.38576532847949],
     [301.0336906326895, 100.16046448916306],
     [238.25686790036065, 70.56535881619311],
     [227.2547443287154, 60.30924933427718],
     [209.13359962247614, 50.817221154818526],
     [203.9561297064078, 50.5813024572758]]
rdst = \
    [[10.822125594094452, 1.42189132706374],
     [21.177065426231174, 1.5297552836484982],
     [25.275895776451954, 1.42189132706374],
     [36.062291434927694, 1.6376192402332563],
     [40.376849698318004, 1.42189132706374],
     [11.900765159942026, -2.1376192402332563],
     [22.25570499207874, -2.1376192402332563],
     [26.785991168638553, -2.029755283648498],
     [37.033067044190524, -2.029755283648498],
     [41.67121717733509, -2.029755283648498]]

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))


def perspective_tform(x, y):
    p1, p2 = tform3_img((x, y))[0]
    return p2, p1


# ***** functions to draw lines *****
def draw_pt(img, x, y, color, sz=2):
    row, col = perspective_tform(x, y)
    row = row * 2
    col = col * 2
    if row >= 0 and row < img.shape[0] * 2 / 2 and col >= 0 and col < img.shape[1] * 2 / 2:
        img[int(row - sz):int(row + sz), int(col - sz):int(col + sz)] = color


def draw_path(img, path_x, path_y, color):
    for x, y in zip(path_x, path_y):
        draw_pt(img, x, y, color)


# ***** functions to draw predicted path *****

def calc_curvature(v_ego, angle_steers, angle_offset=0):
    slip_fator = 0.0014  # slip factor obtained from real data
    steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
    wheel_base = 2.67  # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

    angle_steers_rad = (angle_steers - angle_offset)  # * deg_to_rad
    curvature = angle_steers_rad / (steer_ratio * wheel_base * (1. + slip_fator * v_ego ** 2))
    return curvature

def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
    # *** this function returns the lateral offset given the steering angle, speed and the lookahead distance
    curvature = calc_curvature(v_ego, angle_steers, angle_offset)

    # clip is to avoid arcsin NaNs due to too sharp turns
    y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999)) / 2.)
    return y_actual, curvature


def draw_path_on(img, speed_ms, angle_steers, color=(0, 0, 255)):
    path_x = np.arange(0, 50.1, 0.5)  # 50.1
    path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
    draw_path(img, path_x, path_y, color)


def steering_loop(labels):


    # Create second screen with matplotlibs
    fig = pylab.figure(figsize=[6.4, 1.8], dpi=100)
    ax = fig.gca()
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    line2, = ax.plot([], [], 'r.-', label='Model')
    b = []
    ax.legend(loc='upper left', fontsize=8)

    my_font = pygame.font.SysFont("monospace", 18)
    rand_num_label2 = my_font.render('Model Steer Angle:', 1, red)
    speed_ms = 5

    inputs = []

    start_index = 1000
    for i in range(start_index, start_index + configs.LENGTH):
        data_path = labels[i][0]
        img = cv2.imread(data_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs.append(cv2.resize(img, (224, 224)))

    # Run through all images
    for i in range(configs.LENGTH + start_index, len(labels)):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        img = cv2.imread(labels[i][0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs.pop(0)
        inputs.append(cv2.resize(img, (224, 224)))
        prediction = labels[i][1]
        draw_path_on(img, speed_ms, prediction * -0.1, (255, 0, 0))    # prediction is red

        b.append(prediction)
        line2.set_ydata(b)
        line2.set_xdata(range(len(b)))
        ax.relim()
        ax.autoscale_view()

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        screen.blit(surf, (0, 480))

        # draw on
        pygame.surfarray.blit_array(camera_surface, img.swapaxes(0, 1))
        screen.blit(camera_surface, (0, 0))

        diceDisplay2 = my_font.render(str(prediction * 0.01), 1, red)
        screen.blit(rand_num_label2, (400, 420))
        screen.blit(diceDisplay2, (400, 450))
        clock.tick(60)
        pygame.display.flip()


if __name__ == "__main__":

    labels = pd.read_csv('./main_csv.csv').values
    steering_loop(labels=labels)