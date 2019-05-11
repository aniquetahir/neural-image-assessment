#! python3
import numpy as np
from path import Path

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from utils.nasnet import NASNetMobile
from utils.score_utils import mean_score, std_score

tf.logging.set_verbosity(tf.logging.ERROR)

def nima_nasnet(_dir, progress=False):
    target_size = (224, 224)
    imgs = Path(_dir).files('*.png')
    imgs += Path(_dir).files('*.jpg')
    imgs += Path(_dir).files('*.jpeg')


    with tf.device('/GPU:0'):
        base_model = NASNetMobile((224, 224, 3), include_top=False, pooling='avg', weights=None)
        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)

        model = Model(base_model.input, x)
        model.load_weights('weights/nasnet_weights.h5')

        score_list = []
        total_imgs = len(imgs)
        for i, img_path in enumerate(imgs):
            img = load_img(img_path, target_size=target_size)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            x = preprocess_input(x)

            scores = model.predict(x, batch_size=1, verbose=0)[0]

            mean = mean_score(scores)
            std = std_score(scores)

            file_name = Path(img_path).name.lower()
            score_list.append((file_name, mean, std))

            if progress and i % 100 == 0:
                sys.stdout.write("\r%d/%d" % (i, total_imgs))
                sys.stdout.flush()

            # print("Evaluating : ", img_path)
            # print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))

        return sorted(score_list, key=lambda x: int(x[0].split('.')[0]))


import sys
import json
encoder = json.JSONEncoder()

if __name__ == "__main__":
    dir = sys.argv[1]
    scores = nima_nasnet(dir)
    print(encoder.encode(scores))

