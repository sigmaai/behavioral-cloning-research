#
# Running the validation score
# for the steering models
#
# (c) Yongyang Nie
# 2018, All Rights Reserved
#


from i3d import Inception3D
import configs
from os import path
import pandas as pd
import numpy as np
import helper
import math
import time
import communication


def validation_score(model_path, write_output=False):

    model = Inception3D(input_shape=(configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, configs.CHANNELS),
                        weights_path=model_path)

    # steerings and images
    steering_labels = path.join(configs.VAL_DIR, 'labels.csv')

    df_truth = pd.read_csv(steering_labels, usecols=['frame_id', 'steering_angle'], index_col=None)

    esum = 0
    inputs = []
    predictions = []
    start_time = time.time()

    for i in range(configs.LENGTH):
        file = configs.VAL_DIR + "center/" + str(df_truth['frame_id'].loc[i]) + ".jpg"
        img = helper.load_image(file)
        inputs.append(img)

    # Run through all images
    for i in range(configs.LENGTH, len(df_truth)):

        img = helper.load_image(configs.VAL_DIR + "center/" + str(df_truth['frame_id'].loc[i]) + ".jpg")
        inputs.pop(0)
        inputs.append(img)
        prediction = model.model.predict(np.array([inputs]))[0]
        prediction = prediction[0]
        actual_steers = df_truth['steering_angle'].loc[i]
        e = (actual_steers - prediction) ** 2
        esum += e

        predictions.append(prediction)

        if len(predictions) % 1000 == 0:
            print('.')

    print("time per step: %s seconds" % ((time.time() - start_time) / len(predictions)))

    if write_output:
        print("Writing predictions...")
        pd.DataFrame({"steering_angle": predictions}).to_csv('./result.csv', index=False, header=True)
        print("Done!")
    else:
        print("Not writing outputs")
        print("Done")

    return math.sqrt(esum / len(predictions))


if __name__ == "__main__":

    print("Validating...")
    score = validation_score('i3d_rgb_64_f_v3.h5')
    print("score: " + str(score))

    communication.notify_validation_completion(score, 'i3d_rgb_64_f_v3.h5')
