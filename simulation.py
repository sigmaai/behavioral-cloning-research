#
# Simulation Script
# Neil Nie

import base64  # decoding camera images
import numpy as np
import socketio  # real-time server
import eventlet.wsgi
from PIL import Image  # image manipulation
from flask import Flask  # web framework
from io import BytesIO  # input output
import configs
import cv2
from i3d import Inception3D

# initialize our server
sio = socketio.Server()
# our flask (web) app
app = Flask(__name__)
prev_image_array = None

# set min/max speed for our autonomous car
MAX_SPEED = 10
MIN_SPEED = 10

# and a speed limit
speed_limit = MAX_SPEED
inputs = []


# registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:

        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        cv_image = np.array(image)
        cv_image = cv2.resize(cv_image, (224, 224))
        try:

            if len(inputs) < configs.LENGTH:
                inputs.append(cv_image)
            else:
                inputs.pop(0)
                inputs.append(cv_image)
                prediction = model.model.predict(np.array([inputs]))[0][0]
                steering_angle = -1 * prediction
                # steering_angle = prediction[0]

            print("steering: " + str(steering_angle))

            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow sdown
            else:
                speed_limit = MAX_SPEED

            throttle = 1.0 - (speed / speed_limit) ** 2

            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()},
        skip_sid=True)


if __name__ == '__main__':

    model_path = "./i3d_rgb_64_9.h5"
    model = Inception3D(input_shape=(configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, configs.CHANNELS),
                        weights_path=model_path)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)