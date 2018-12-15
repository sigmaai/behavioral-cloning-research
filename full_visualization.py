"""
Dataset visualization tool
Original By: Comma.ai and Chris Gundling
Revised and used by Neil Nie
"""

import matplotlib.backends.backend_agg as agg
import numpy as np
import pandas as pd
import cv2
import pygame
import pylab
import helper
import configs
from termcolor import colored
from PIL import Image as PILImage
from PIL import ImageFont
from PIL import ImageDraw

from i3d import Inception3D

pygame.init()
size = (640, 640)
pygame.display.set_caption("Behavioral Cloning Viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
screen.set_alpha(None)

camera_surface = pygame.surface.Surface((640, 480), 0, 24).convert()
clock = pygame.time.Clock()

PI_RAD = (180 / np.pi)
white = (255, 255, 255)
red = (255, 102, 102)
blue = (102, 178, 255)
black = (0, 0, 0)

# Create second screen with matplotlibs
fig = pylab.figure(figsize=[6.4, 1.6], dpi=100)
ax = fig.gca()
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
line1, = ax.plot([], [], 'b.-', label='Model')
line2, = ax.plot([], [], 'r.-', label='Human')
a = []
b = []
ax.legend(loc='upper left', fontsize=8)

myFont = pygame.font.SysFont("monospace", 18)
static_label1 = myFont.render('Model Prediction:', 1, white)
static_label2 = myFont.render('Ground Truth:', 1, white)
static_label3 = myFont.render('Abs. Error', 1, white)


def visualize_steering_wheel(image, angle):
    background = PILImage.fromarray(np.uint8(image))
    sw = PILImage.open("./media/sw.png")
    sw = sw.rotate(angle * PI_RAD)
    sw = sw.resize((80, 80), PILImage.ANTIALIAS)
    background.paste(sw, (10, 55), sw)

    draw = ImageDraw.Draw(background)
    font = ImageFont.truetype("./media/FiraMono-Medium.otf", 18)
    draw.text((80, 200), str(round(angle, 3)), (255, 255, 255), font=font)
    steering_img = cv2.resize(np.array(background), (640, 480))
    return steering_img


def pygame_loop(label, prediction, img):

    angle = prediction[0] * 5
    accel = prediction[1] * 1
    if accel < 0:
        pred_label = myFont.render('Slowing', 1, white)
    else:
        pred_label = myFont.render('Speeding', 1, white)
    gt_label = myFont.render(str(label), 1, red)
    pred_val = myFont.render(str(accel), 1, blue)
    error_label = myFont.render(str(abs(round((accel - label), 3))), 1, white)

    img = visualize_steering_wheel(image=img, angle=angle)

    a.append(accel)         # a is prediction
    b.append(label)         # b is label
    line1.set_ydata(a)
    line1.set_xdata(range(len(a)))
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
    screen.blit(static_label3, (15, 15))
    screen.blit(error_label, (15, 30))
    screen.blit(static_label1, (50, 420))
    screen.blit(pred_label, (280, 420))
    screen.blit(pred_val, (50, 450))

    screen.blit(static_label2, (450, 420))
    screen.blit(gt_label, (450, 450))

    clock.tick(60)
    pygame.display.flip()


def visualize_accel(model_path, label_path):

    print(colored('Preparing', 'blue'))

    model = Inception3D(weights_path=model_path, input_shape=(configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, 3))

    # read the steering labels and image path
    labels = pd.read_csv(label_path).values

    inputs = []
    starting_index = 9000
    end_index = 0
    # init_speed = labels[starting_index][2]

    for i in range(starting_index, starting_index + configs.LENGTH):
        img = helper.load_image(configs.TRAIN_DIR + "frame" + str(i) + ".jpg")
        inputs.append(img)

    print(colored('Started', 'blue'))

    # Run through all images
    for i in range(starting_index + configs.LENGTH + 1, len(labels) - 1 - end_index):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        img = helper.load_image("/hdd/ssd_2/dataset/speedchallenge/data/train/" + str(labels[i][1]), auto_resize=False)
        in_frame = cv2.resize(img, (configs.IMG_WIDTH, configs.IMG_HEIGHT))
        inputs.pop(0)
        inputs.append(in_frame)
        cmds = model.model.predict(np.array([np.asarray(inputs)]))[0]
        label_accel = (labels[i][2] - labels[i - 1][2]) * 20
        pygame_loop(label=label_accel, prediction=cmds, img=img)


if __name__ == "__main__":

    visualize_accel(label_path='/hdd/ssd_2/dataset/speedchallenge/data/data.csv',
                    model_path='i3d_rgb_64_v10.h5')
