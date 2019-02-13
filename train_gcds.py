#!/usr/bin/python
#
# training file for i3d model, golf cart dataset
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

    # load_weights = "./i3d_small_v1.h5"
    save_weights = "./i3d_vehicle_dataset_1.h5"

    i3d_model = Inception3D(input_shape=(configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, 3))
                            # weights_path=load_weights)
    i3d_model.summary()

    labels = pd.read_csv('./main_csv.csv').values
    val_labels = pd.read_csv('/home/neil/dataset/steering/test/labels.csv')

    i3d_model.train(type='gc_rgb', labels=labels, val_labels=val_labels,
                    epochs=5, epoch_steps=1000,
                    validation=False, val_steps=500,
                    save_path=save_weights)

    communication.notify_training_completion(configs.LOG_PATH)