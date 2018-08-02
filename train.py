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

from i3d import i3d
import configs
import pandas as pd
import communication


if __name__ == '__main__':


    i3d_model = i3d(input_shape=(configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, 3),
                    weights_path="./i3d_rgb_64_6.h5")
    i3d_model.summary()

    labels = pd.read_csv('/home/neil/dataset/udacity/main.csv').values
    val_labels = pd.read_csv('/home/neil/dataset/steering/test/labels.csv')

    i3d_model.train(type='rgb', labels=labels, val_labels=val_labels, epochs=3, epoch_steps=800, validation=True, val_steps=1000,
                    save_path='./i3d_rgb_64_7.h5', log_path=configs.LOG_PATH)

    communication.notify_training_completion(configs.LOG_PATH)