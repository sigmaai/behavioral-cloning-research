#!/usr/bin/python
#
# training file for i3d model.
# this project supports the udacity self-driving dataset.
# link: https://github.com/udacity/self-driving-car/tree/master/datasets
#
#
# Author: Neil (Yongyang) Nie
# Copyright: (c) 2018
# Licence: MIT
# Contact: contact@neilnie.com
#

from i3d import Inception3D
import configs
import pandas as pd
import communication


if __name__ == '__main__':

    load_weights = "./i3d_rgb_64_8.h5"
    save_weights = "./i3d_rgb_64_9.h5"

    i3d_model = Inception3D(input_shape=(configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, 3),
                            weights_path=load_weights)
    i3d_model.summary()

    labels = pd.read_csv('/home/neil/dataset/udacity/main.csv').values
    val_labels = pd.read_csv('/home/neil/dataset/steering/test/labels.csv')

    i3d_model.train(type='rgb', labels=labels, val_labels=val_labels, epochs=5,
                    epoch_steps=300, validation=True, val_steps=500,
                    save_path=save_weights,
                    log_path=configs.LOG_PATH)

    communication.notify_training_completion(configs.LOG_PATH)