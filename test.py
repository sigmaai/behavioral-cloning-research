#!/usr/bin/python
#
# training file for i3d model.
# this project supports the udacity self-driving dataset.
# link: https://github.com/udacity/self-driving-car/tree/master/datasets
#
# will soon support Common AI speed dataset.
#
# Author: Neil (Yongyang) Nie
# Copyright: (c) 2018
# Licence: MIT
# Contact: contact@neilnie.com
#

from i3d import i3d
import configs
import helper
import pandas as pd

if __name__ == '__main__':

    i3d_model = i3d(input_shape=(configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, configs.CHANNELS),
                    ) # weights_path='./i3d_32_11.h5'
    # i3d_model.summary()

    labels = pd.read_csv('/home/neil/dataset/udacity/main.csv').values
    val_label = pd.read_csv('/home/neil/dataset/steering/test/labels.csv')
    train_gen = helper.udacity_batch_generator(batch_size=1, data=labels, augment=False)
    val_gen = helper.validation_batch_generator(batch_size=1, data=val_label)

    i3d_model.train(train_gen=train_gen, epochs=10, epoch_steps=3000, val_gen=val_gen, val_steps=1000, save_path='./i3d_speed_u_32_1.h5')
