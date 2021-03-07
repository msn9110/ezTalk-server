import json
import os
import numpy as np
# Keras
import tensorflow as tf
from config import general_data_path, get_settings
from tensorflow.keras.models import load_model

from take_audio import get_recognized_syllable_lists


def take_probs(stn='大家好'):
    return get_recognized_syllable_lists(stn, -1, 'msn9110')[-1]


# zhuyin
def read_labels(filename):
    with open(filename) as f:
        available = list(filter(lambda _: bool(_), f.read().split('\n')))
    return available


labels = ['_silence_'] + read_labels(general_data_path['valid_zhuyin']) + ['_unknown_']
label_index = {l: i for i, l in enumerate(labels)}
index_label = {i: l for l, i in label_index.items()}


def get_reranking_syllable_lists(user, recognized_syllable_lists=None):
    if not isinstance(recognized_syllable_lists, list) or not recognized_syllable_lists:
        raise Exception('Type error or empty list')
    uid, (*_, mdir, _), *_ = get_settings(user)

    info_path = os.path.join(mdir, 'transformer.json')
    with open(info_path) as f:
        path = json.load(f)['path']

    # load trained model
    attention_l = tf.keras.layers.Attention()
    model = load_model(path, custom_objects={'Attention': attention_l})

    print('use transformer')

    max_len = 100

    silence_v = [0.] * len(label_index)
    silence_v[label_index['_silence_']] = 1.
    
    size = len(recognized_syllable_lists)
    vectors = []

    # make input
    for i, pron_v in enumerate(recognized_syllable_lists):
        vec_in = [0.] * len(label_index)
        for label, pr in pron_v:
            vec_in[label_index[label]] = pr
        vectors.append(vec_in)

    if size < max_len:
        offset = max_len - size
        vectors = [silence_v for _ in range(offset)] + vectors
    X_in = np.array(vectors)

    predicts = model.predict(x=[X_in, X_in])[0]
    
    predict_ls = []
    for pred in predicts:
        pred_list = []
        for i, pr in enumerate(pred):
            zhuyin = index_label[i]
            pred_list.append((zhuyin, pr))
        pred_list.sort(key=lambda x: x[1], reverse=True)
        if not pred_list[0][0].startswith('_'):
            predict_ls.append(pred_list)
        
    return predict_ls


if __name__ == '__main__':
    pl = take_probs('你好')
    get_reranking_syllable_lists(pl)
