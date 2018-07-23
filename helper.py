#
# training helper
# methods for training deep learning models
# By Neil Nie
# (c) Yongyang Nie, 2018. All Rights Reserved
# Contact: contact@neilnie.com

import cv2
import numpy as np
import random
import configs


def rotate(img):
    row, col, channel = img.shape
    angle = np.random.uniform(-15, 15)
    rotation_point = (row / 2, col / 2)
    rotation_matrix = cv2.getRotationMatrix2D(rotation_point, angle, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (col, row))
    return rotated_img


def blur(img):
    r_int = np.random.randint(0, 2)
    odd_size = 2 * r_int + 1
    return cv2.GaussianBlur(img, (odd_size, odd_size), 0)


def random_shadow(image):
    """
    Generates and adds random shadow
    """

    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = configs.IMG_WIDTH * np.random.rand(), 0
    x2, y2 = configs.IMG_WIDTH * np.random.rand(), configs.IMG_HEIGHT
    xm, ym = np.mgrid[0:configs.IMG_HEIGHT, 0:configs.IMG_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def flip_image(image):

    horizontal_img = cv2.flip(image, 0)
    return horizontal_img


def augument(images, angle):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)

    :param images: a list of images waiting to be augmented
    :param angle: the angle associated with that image
    :returns: a tensor of the augmented images
    :returns: the steering angle
    """
    a = np.random.randint(0, 3, [1, 5]).astype('bool')[0]
    if a[0] == 1:
        for idx in range(len(images)):
            image = images[idx].astype(np.uint8)
            images[idx] = random_shadow(image)
    if a[1] == 1:
        for idx in range(len(images)):
            image = images[idx].astype(np.uint8)
            images[idx] = blur(image)
    if a[2] == 1:
        for idx in range(len(images)):
            image = images[idx].astype(np.uint8)
            images[idx] = random_brightness(image)
    # if a[3] == 1:
    #     for idx in range(len(images)):
    #         image = images[idx].astype(np.uint8)
    #         images[idx] = flip_image(image)
    #     angle = -1 * angle

    return images, angle


def get_output_dim(model):
    """
    Infer output dimension from model by inspecting its layers.

    :param model - tensorflow model
    :returns - output dimension
    """
    for layer in reversed(model.layers):
        if hasattr(layer, 'output_shape'):
            return layer.output_shape[-1]

    raise ValueError('Could not infer output dim')


def load_image(image_file, auto_resize=True):

    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if auto_resize:
        img = cv2.resize(img, (configs.IMG_WIDTH, configs.IMG_HEIGHT))

    return img


def load_gray_image(image_file):

    img = cv2.imread(image_file)
    img = cv2.resize(img, (configs.IMG_WIDTH, configs.IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def udacity_batch_generator(data, batch_size, augment):

    """
    Generate training images given image paths and associated steering angles

    Args:
         data (numpy.array)        : the loaded data (converted to list from pandas format)
         batch_size (int)   : batch size for training
         training: (boolean): whether to use augmentation or not.

    Yields:
         images ([tensor])  : images for training
         angles ([float])   : the corresponding steering angles


    """

    images = np.empty([batch_size, configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, 3], dtype=np.int32)
    labels = np.empty([batch_size])

    while True:

        c = 0

        for index in np.random.permutation(data.shape[0]):

            imgs = np.empty([configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, 3], dtype=np.int32)

            if index < configs.LENGTH:
                start = 0
                end = configs.LENGTH
            elif index + configs.LENGTH >= len(data):
                start = len(data) - configs.LENGTH - 1
                end = len(data) - 1
            else:
                start = index
                end = index + configs.LENGTH

            for i in range(start, end):
                center_path = str(data[i][5])
                image = load_image(center_path)
                imgs[i - start] = image

            # augmentaion if needed
            if augment and bool(random.getrandbits(1)):
                imgs, angle = augument(imgs, data[end][6])
            else:
                angle = data[end][6]

            images[c] = imgs
            labels[c] = angle

            c += 1

            if c == batch_size:
                break

        yield images, labels


def validation_batch_generator(data, batch_size):

    """
    Generate training image give image paths and associated steering angles
    """

    images = np.empty([batch_size, configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, 3], dtype=np.int32)
    labels = np.empty([batch_size])

    while True:

        c = 0

        for index in np.random.permutation(data.shape[0]):

            imgs = np.empty([configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, 3], dtype=np.int32)

            if index < configs.LENGTH:
                start = 0
                end = configs.LENGTH
            elif index + configs.LENGTH >= len(data):
                start = len(data) - configs.LENGTH - 1
                end = len(data) - 1
            else:
                start = index
                end = index + configs.LENGTH

            for i in range(start, end):
                center_path = '/home/neil/dataset/steering/test/center/' + str(data['frame_id'].loc[i]) + ".jpg"
                image = load_image(center_path)
                imgs[i - start] = image

            images[c] = imgs
            labels[c] = data['steering_angle'].loc[end]

            c += 1

            if c == batch_size:
                break

        yield images, labels


