"""Inception-v1 Inflated 3D ConvNet used for Kinetics CVPR paper.

The model is introduced in:

Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
Joao Carreira, Andrew Zisserman
https://arxiv.org/abs/1705.07750v1

Initially written by Ese dlpbc
Modified & improved by Neil Nie.

MIT Licence. (c) Yongyang Nie, 2018 All Rights Reserved
Contact: contact@neilnie.com

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings


from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv3D
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.layers import Dropout
from keras.layers import Reshape
from keras.optimizers import Adam
from keras.layers import Flatten
from keras.callbacks import TensorBoard
import datetime
from keras import backend as K


class i3d:

    def __init__(self, weights_path=None, input_shape=None,
                 dropout_prob=0.0, endpoint_logit=True, classes=1):

        # type: (weights_path, input_shape, dropout_prob, endpoint_logit, classes) -> None

        '''Instantiates the Inflated 3D Inception v1 architecture.

        Optionally loads weights pre-trained on Kinetics. Note that when using TensorFlow,
        Always channel last. The model and the weights are compatible with both
        TensorFlow. The data format convention used by the model is the one
        specified in your Keras config file.
        Note that the default input frame(image) size for this model is 224x224.

        :param weights_path: one of `None` (random initialization)
        :param input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape should have exactly
                3 inputs channels. NUM_FRAMES should be no smaller than 8. The authors
                used 64 frames per example for training and testing on kinetics dataset
                Width and height should be no smaller than 32.
                i.e.: `(64, 150, 150, 3)` would be one valid value.
        :param dropout_prob: optional, dropout probability applied in dropout layer
                after global average pooling layer.
                0.0 means no dropout is applied, 1.0 means dropout is applied to all features.
                Note: Since Dropout is applied just before the classification
                layer, it is only useful when `include_top` is set to True.
        :param endpoint_logit: (boolean) optional. If True, the model's forward pass
                will end at producing logits. Otherwise, softmax is applied after producing
                the logits to produce the class probabilities prediction. Setting this parameter
                to True is particularly useful when you want to combine results of rgb model
                and optical flow model.
                - `True` end model forward pass at logit output
                - `False` go further after logit to produce softmax predictions
                Note: This parameter is only useful when `include_top` is set to True.
        :param classes: For regression (i.e. behavorial cloning) 1 is the default value.
                optional number of classes to classify images into, only to be specified
                if `include_top` is True, and if no `weights` argument is specified.

        '''

        self.input_shape = input_shape
        self.dropout_prob = dropout_prob
        self.endpoint_logit = endpoint_logit
        self.classes = classes
        self.weight_path = weights_path

        input_shape = self._obtain_input_shape(self.input_shape, default_frame_size=224,
                                               min_frame_size=32, default_num_frames=64,
                                               min_num_frames=8, data_format=K.image_data_format())  # weights=weights

        img_input = Input(shape=input_shape)
        self.model = self.create_model(img_input)

        if weights_path:
            self.model.load_weights(weights_path)
            print("loaded weights:" + weights_path)

    def summary(self):
        print(self.model.summary())

    def train(self, train_gen, epochs=10, epoch_steps=5000, val_gen=None, val_steps=None, validation=False, log_path="logs/32/", save_path=None):

        '''training the model

        :param train_gen: training generator. For details, please read the implementation in helper.py
        :param val_gen: validation generator, for now it's required.
        :param epoch: number of training epochs.
        :param epoch_steps: number of training steps per epoch. (!= batch_size)
        :param val_steps: number of validation steps
        :param log_path: training log path.
        :param validation: run validation or not. If not validating, val_gen and val_steps can be non.
        '''

        if save_path == None:
            print("[WARNING]: trained model will not be saved. Please specify save_path")

        tensorboard = TensorBoard(log_dir=(log_path + "/{}".format(datetime.datetime.now())))

        if validation:
            if val_gen and val_steps:
                self.model.fit_generator(train_gen, steps_per_epoch=epoch_steps,
                                         epochs=epochs, validation_data=val_gen,
                                         validation_steps=val_steps,
                                         verbose=1, callbacks=[tensorboard])  #
            else:
                raise Exception('please specify val_gen and val_steps')

        else:
            self.model.fit_generator(train_gen, steps_per_epoch=epoch_steps, epochs=epochs, verbose=1, callbacks=[tensorboard])

        self.model.save(save_path)

    def create_model(self, img_input):

        '''create and return the i3d model
        :param: img_input: input shape of the network.
        :return: A Keras model instance.
        '''

        # Determine proper input shape

        channel_axis = 4

        # Downsampling via convolution (spatial and temporal)
        x = self.conv3d_bath_norm(img_input, 64, 7, 7, 7, strides=(2, 2, 2), padding='same', name='Conv3d_1a_7x7')

        # Downsampling (spatial only)
        x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_2a_3x3')(x)
        x = self.conv3d_bath_norm(x, 64, 1, 1, 1, strides=(1, 1, 1), padding='same', name='Conv3d_2b_1x1')
        x = self.conv3d_bath_norm(x, 192, 3, 3, 3, strides=(1, 1, 1), padding='same', name='Conv3d_2c_3x3')

        # Downsampling (spatial only)
        x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_3a_3x3')(x)

        # Mixed 3b
        branch_0 = self.conv3d_bath_norm(x, 64, 1, 1, 1, padding='same', name='Conv3d_3b_0a_1x1')

        branch_1 = self.conv3d_bath_norm(x, 96, 1, 1, 1, padding='same', name='Conv3d_3b_1a_1x1')
        branch_1 = self.conv3d_bath_norm(branch_1, 128, 3, 3, 3, padding='same', name='Conv3d_3b_1b_3x3')

        branch_2 = self.conv3d_bath_norm(x, 16, 1, 1, 1, padding='same', name='Conv3d_3b_2a_1x1')
        branch_2 = self.conv3d_bath_norm(branch_2, 32, 3, 3, 3, padding='same', name='Conv3d_3b_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3b_3a_3x3')(x)
        branch_3 = self.conv3d_bath_norm(branch_3, 32, 1, 1, 1, padding='same', name='Conv3d_3b_3b_1x1')

        x = layers.concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis, name='Mixed_3b')

        # Mixed 3c
        branch_0 = self.conv3d_bath_norm(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_0a_1x1')

        branch_1 = self.conv3d_bath_norm(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_1a_1x1')
        branch_1 = self.conv3d_bath_norm(branch_1, 192, 3, 3, 3, padding='same', name='Conv3d_3c_1b_3x3')

        branch_2 = self.conv3d_bath_norm(x, 32, 1, 1, 1, padding='same', name='Conv3d_3c_2a_1x1')
        branch_2 = self.conv3d_bath_norm(branch_2, 96, 3, 3, 3, padding='same', name='Conv3d_3c_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3c_3a_3x3')(x)
        branch_3 = self.conv3d_bath_norm(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_3c_3b_1x1')

        x = layers.concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis, name='Mixed_3c')

        # Downsampling (spatial and temporal)
        x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='MaxPool2d_4a_3x3')(x)

        # Mixed 4b
        branch_0 = self.conv3d_bath_norm(x, 192, 1, 1, 1, padding='same', name='Conv3d_4b_0a_1x1')

        branch_1 = self.conv3d_bath_norm(x, 96, 1, 1, 1, padding='same', name='Conv3d_4b_1a_1x1')
        branch_1 = self.conv3d_bath_norm(branch_1, 208, 3, 3, 3, padding='same', name='Conv3d_4b_1b_3x3')

        branch_2 = self.conv3d_bath_norm(x, 16, 1, 1, 1, padding='same', name='Conv3d_4b_2a_1x1')
        branch_2 = self.conv3d_bath_norm(branch_2, 48, 3, 3, 3, padding='same', name='Conv3d_4b_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4b_3a_3x3')(x)
        branch_3 = self.conv3d_bath_norm(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4b_3b_1x1')

        x = layers.concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis, name='Mixed_4b')

        # Mixed 4c
        branch_0 = self.conv3d_bath_norm(x, 160, 1, 1, 1, padding='same', name='Conv3d_4c_0a_1x1')

        branch_1 = self.conv3d_bath_norm(x, 112, 1, 1, 1, padding='same', name='Conv3d_4c_1a_1x1')
        branch_1 = self.conv3d_bath_norm(branch_1, 224, 3, 3, 3, padding='same', name='Conv3d_4c_1b_3x3')

        branch_2 = self.conv3d_bath_norm(x, 24, 1, 1, 1, padding='same', name='Conv3d_4c_2a_1x1')
        branch_2 = self.conv3d_bath_norm(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4c_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4c_3a_3x3')(x)
        branch_3 = self.conv3d_bath_norm(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4c_3b_1x1')

        x = layers.concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis, name='Mixed_4c')

        # Mixed 4d
        branch_0 = self.conv3d_bath_norm(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_0a_1x1')

        branch_1 = self.conv3d_bath_norm(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_1a_1x1')
        branch_1 = self.conv3d_bath_norm(branch_1, 256, 3, 3, 3, padding='same', name='Conv3d_4d_1b_3x3')

        branch_2 = self.conv3d_bath_norm(x, 24, 1, 1, 1, padding='same', name='Conv3d_4d_2a_1x1')
        branch_2 = self.conv3d_bath_norm(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4d_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4d_3a_3x3')(x)
        branch_3 = self.conv3d_bath_norm(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4d_3b_1x1')

        x = layers.concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis, name='Mixed_4d')

        # Mixed 4e
        branch_0 = self.conv3d_bath_norm(x, 112, 1, 1, 1, padding='same', name='Conv3d_4e_0a_1x1')

        branch_1 = self.conv3d_bath_norm(x, 144, 1, 1, 1, padding='same', name='Conv3d_4e_1a_1x1')
        branch_1 = self.conv3d_bath_norm(branch_1, 288, 3, 3, 3, padding='same', name='Conv3d_4e_1b_3x3')

        branch_2 = self.conv3d_bath_norm(x, 32, 1, 1, 1, padding='same', name='Conv3d_4e_2a_1x1')
        branch_2 = self.conv3d_bath_norm(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4e_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4e_3a_3x3')(x)
        branch_3 = self.conv3d_bath_norm(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4e_3b_1x1')

        x = layers.concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis, name='Mixed_4e')

        # Mixed 4f
        branch_0 = self.conv3d_bath_norm(x, 256, 1, 1, 1, padding='same', name='Conv3d_4f_0a_1x1')

        branch_1 = self.conv3d_bath_norm(x, 160, 1, 1, 1, padding='same', name='Conv3d_4f_1a_1x1')
        branch_1 = self.conv3d_bath_norm(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_4f_1b_3x3')

        branch_2 = self.conv3d_bath_norm(x, 32, 1, 1, 1, padding='same', name='Conv3d_4f_2a_1x1')
        branch_2 = self.conv3d_bath_norm(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_4f_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4f_3a_3x3')(x)
        branch_3 = self.conv3d_bath_norm(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_4f_3b_1x1')

        x = layers.concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis, name='Mixed_4f')

        # Downsampling (spatial and temporal)
        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='MaxPool2d_5a_2x2')(x)

        # Mixed 5b
        branch_0 = self.conv3d_bath_norm(x, 256, 1, 1, 1, padding='same', name='Conv3d_5b_0a_1x1')

        branch_1 = self.conv3d_bath_norm(x, 160, 1, 1, 1, padding='same', name='Conv3d_5b_1a_1x1')
        branch_1 = self.conv3d_bath_norm(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_5b_1b_3x3')

        branch_2 = self.conv3d_bath_norm(x, 32, 1, 1, 1, padding='same', name='Conv3d_5b_2a_1x1')
        branch_2 = self.conv3d_bath_norm(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5b_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5b_3a_3x3')(x)
        branch_3 = self.conv3d_bath_norm(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5b_3b_1x1')

        x = layers.concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis, name='Mixed_5b')

        # Mixed 5c
        branch_0 = self.conv3d_bath_norm(x, 384, 1, 1, 1, padding='same', name='Conv3d_5c_0a_1x1')

        branch_1 = self.conv3d_bath_norm(x, 192, 1, 1, 1, padding='same', name='Conv3d_5c_1a_1x1')
        branch_1 = self.conv3d_bath_norm(branch_1, 384, 3, 3, 3, padding='same', name='Conv3d_5c_1b_3x3')

        branch_2 = self.conv3d_bath_norm(x, 48, 1, 1, 1, padding='same', name='Conv3d_5c_2a_1x1')
        branch_2 = self.conv3d_bath_norm(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5c_2b_3x3')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5c_3a_3x3')(x)
        branch_3 = self.conv3d_bath_norm(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5c_3b_1x1')

        x = layers.concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis, name='Mixed_5c')

        # Classification block
        x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)
        x = Dropout(self.dropout_prob)(x)

        x = self.conv3d_bath_norm(x, self.classes, 1, 1, 1, padding='same', use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

        num_frames_remaining = int(x.shape[1])
        x = Reshape((num_frames_remaining, self.classes))(x)
        x = Flatten()(x)
        x = Dense(self.classes)(x)

        # logits (raw scores for each class)
        # x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False), output_shape=lambda s: (s[0], s[2]))(x)

        # if not self.endpoint_logit:
        #     x = Activation('softmax', name='prediction')(x)
        # ===

        inputs = img_input

        # create model
        model = Model(inputs, x, name='i3d_inception')
        optimizer = Adam(lr=1e-5, decay=1e-6)
        #
        model.compile(loss=self.root_mean_squared_error, optimizer=optimizer, metrics=[self.root_mean_squared_error])

        return model

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    @staticmethod
    def conv3d_bath_norm(x, filters, num_frames, num_row, num_col, padding='same', strides=(1, 1, 1),
                         use_bias=False, use_activation_fn=True, use_bn=True, name=None):

        '''

        :param x: input tensor.
        :param filters: filters in `Conv3D`.
        :param num_frames: frames (time depth) of the convolution kernel.
        :param num_row: height of the convolution kernel.
        :param num_col: width of the convolution kernel.
        :param padding: padding mode in `Conv3D`.
        :param strides: strides in `Conv3D`.
        :param use_bias: use bias or not
        :param use_activation_fn: use an activation function or not.
        :param use_bn: use batch normalization or not.
        :param name: name of the ops; will become `name + '_conv'`
                for the convolution and `name + '_bn'` for the
                batch norm layer.
        :return: Output tensor after applying `Conv3D` and `BatchNormalization`.
        '''

        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv3D(filters, (num_frames, num_row, num_col), strides=strides, padding=padding, use_bias=use_bias,
                   name=conv_name)(x)

        if use_bn:
            bn_axis = 4
            x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

        if use_activation_fn:
            x = Activation('relu', name=name)(x)

        return x

    @staticmethod
    def _obtain_input_shape(input_shape, default_frame_size, min_frame_size, default_num_frames,
                            min_num_frames, data_format, weights=None):

        """Internal utility to compute/validate the model's input shape.
        (Adapted from `keras/applications/imagenet_utils.py`)

        # Arguments
            input_shape: either None (will return the default network input shape),
                or a user-provided shape to be validated.
            default_frame_size: default input frames(images) width/height for the model.
            min_frame_size: minimum input frames(images) width/height accepted by the model.
            default_num_frames: default input number of frames(images) for the model.
            min_num_frames: minimum input number of frames accepted by the model.
            data_format: image data format to use.
            require_flatten: whether the model is expected to
                be linked to a classifier via a Flatten layer.
            weights: one of `None` (random initialization)
                or 'kinetics_only' (pre-training on Kinetics dataset).
                or 'imagenet_and_kinetics' (pre-training on ImageNet and Kinetics datasets).
                If weights='kinetics_only' or weights=='imagenet_and_kinetics' then
                input channels must be equal to 3.

        # Returns
            An integer shape tuple (may include None entries).

        # Raises
            ValueError: in case of invalid argument values.
        """
        if weights != 'kinetics_only' and weights != 'imagenet_and_kinetics' and input_shape and len(input_shape) == 4:
            if data_format == 'channels_first':
                if input_shape[0] not in {1, 3}:
                    warnings.warn(
                        'This model usually expects 1 or 3 input channels. '
                        'However, it was passed an input_shape with ' +
                        str(input_shape[0]) + ' input channels.')
                default_shape = (input_shape[0], default_num_frames, default_frame_size, default_frame_size)
            else:
                if input_shape[-1] not in {1, 3}:
                    warnings.warn(
                        'This model usually expects 1 or 3 input channels. '
                        'However, it was passed an input_shape with ' +
                        str(input_shape[-1]) + ' input channels.')
                default_shape = (default_num_frames, default_frame_size, default_frame_size, input_shape[-1])
        else:
            default_shape = (default_num_frames, default_frame_size, default_frame_size, 3)

        if (weights == 'kinetics_only' or weights == 'imagenet_and_kinetics'):
            if input_shape is not None:
                if input_shape != default_shape:
                    raise ValueError('When setting`include_top=True` '
                                     'and loading `imagenet` weights, '
                                     '`input_shape` should be ' +
                                     str(default_shape) + '.')
            return default_shape

        if input_shape:

            if input_shape is not None:
                if len(input_shape) != 4:
                    raise ValueError('`input_shape` must be a tuple of four integers.')
                if input_shape[-1] != 3 and (weights == 'kinetics_only' or weights == 'imagenet_and_kinetics'):
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')

                if input_shape[0] is not None and input_shape[0] < min_num_frames:
                    raise ValueError('Input number of frames must be at least ' +
                                     str(min_num_frames) + '; got '
                                                           '`input_shape=' + str(input_shape) + '`')

                if ((input_shape[1] is not None and input_shape[1] < min_frame_size) or
                        (input_shape[2] is not None and input_shape[2] < min_frame_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_frame_size) + 'x' + str(min_frame_size) + '; got '
                                                                                       '`input_shape=' + str(input_shape) + '`')

        else:
            input_shape = default_shape

        return input_shape


