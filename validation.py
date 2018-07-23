#
# (c) Yongyang Nie
# 2018, All Rights Reserved
#


from i3d import i3d
import configs
from os import path
import pandas as pd
import numpy as np
import helper
import math
import time


def validation_score(model_path, write_output=False):

    model = i3d(input_shape=(configs.LENGTH, configs.IMG_HEIGHT, configs.IMG_WIDTH, configs.CHANNELS), weights_path=model_path)

    # steerings and images
    steering_labels = path.join(configs.VAL_DIR, 'labels.csv')

    df_truth = pd.read_csv(steering_labels, usecols=['frame_id', 'steering_angle'], index_col=None)

    esum = 0
    count = 0
    input = []
    predictions = []

    start_time = time.time()

    for i in range(configs.LENGTH):
        file = configs.VAL_DIR + "center/" + str(df_truth['frame_id'].loc[i]) + ".jpg"
        img = helper.load_image(file)
        input.append(img)

    # Run through all images
    for i in range(configs.LENGTH, len(df_truth)):

        img = helper.load_image(configs.VAL_DIR + "center/" + str(df_truth['frame_id'].loc[i]) + ".jpg")
        input.pop(0)
        input.append(img)
        input_array = np.array([input])
        prediction = model.model.predict(input_array)[0][0]
        actual_steers = df_truth['steering_angle'].loc[i]
        e = (actual_steers - prediction) ** 2
        esum += e
        count += 1

        predictions.append(prediction)

        if count % 1000 == 0:
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
    score = validation_score('i3d_32_15.h5')
    print("score: " + str(score))
