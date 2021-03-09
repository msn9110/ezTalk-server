import numpy as np
from six.moves import xrange
import sys
sys.path.append('../')

from cmodels import which_set, duplicate_data, general_data_path
from phoneme import indexes, zindexes, rev_zindexes, optionals, valid_zhuyins

epsilon = 1e-10


def _normalize(prob_l):
    total = sum(prob_l)
    return [pr / max(1.0, total) + epsilon for pr in prob_l]


def _partial_normalize(_vec):
    out = []
    for _ in _vec:
        out += _normalize(_)
    return out


def make_input(data):
    start = 1 if len(data) == 3 else 4
    out = [[0.0] * len(indexes[start + i])
           for i in range(len(data))]

    for i, l in enumerate(data):
        d = indexes[start + i]
        for w, pr in l:
            if w in d:
                j = d[w]
                out[i][j] = float(pr)
    return _partial_normalize(out)


def make_record(inputs, label):
    vector = make_input(inputs)
    if label not in zindexes:
        return None
    ground_truth_i = zindexes[label]
    ground_truth_v = [0.0] * len(zindexes)
    ground_truth_v[ground_truth_i] = 1
    return [np.array(vector), np.array(ground_truth_v)]


class DataProcessor:

    balance_data = False
    def __init__(self, json_path, validation_percentage,
                 testing_percentage, test_set_json_path=None):
        self.data = {}
        self.prepare_data_index(json_path, validation_percentage, testing_percentage,
                                test_set_json_path)

    def prepare_data_index(self, json_path, validation_percentage, testing_percentage,
                           test_set_json_path=None):
        self.data_index = {'validation': [], 'testing': [], 'training': []}

        import json

        with open(json_path) as f:
            data = json.load(f)

        flag = True
        for filename, v in data.items():
            d_index = which_set(filename, validation_percentage, testing_percentage)
            in_, ground_truth = v
            flag = not (len(in_) == 2)
            rec = make_record(in_, ground_truth)
            if not rec:
                continue
            self.data[filename] = rec
            self.data_index[d_index].append({'file': filename, 'label': ground_truth})

        # include valid pinyin to training set
        if flag and False:
            with open(general_data_path['pinyin']) as f:
                data = json.load(f)
            d_index = 'training'

            st = len(indexes[1])
            sm = len(indexes[2])

            for filename, v in data.items():

                in_, ground_truth = v
                in_ = [in_[:st], in_[st:st + sm], in_[st + sm:]]
                rec = make_record(in_, ground_truth)
                if not rec:
                    continue
                self.data[filename] = rec
                self.data_index[d_index].append({'file': filename, 'label': ground_truth})

        if self.balance_data:
            self.data_index['training'] = duplicate_data(self.data_index['training'])
            self.data_index['validation'] = duplicate_data(self.data_index['validation'])

        if test_set_json_path:
            test_set = []
            with open(test_set_json_path) as f:
                data = json.load(f)

            for filename, v in data.items():
                in_, ground_truth = v
                self.data[filename] = make_record(in_, ground_truth)
                test_set.append({'file': filename, 'label': ground_truth})
            if test_set:
                self.data_index['testing'] = test_set
        print(self.set_size('training'))
        print(self.set_size('validation'))
        print(self.set_size('testing'))
        #exit()

    def set_size(self, mode):
        """Calculates the number of samples in the dataset partition.

        Args:
          mode: Which partition, must be 'training', 'validation', or 'testing'.

        Returns:
          Number of samples in the partition.
        """
        return len(self.data_index[mode])

    def get_data(self, how_many, offset, mode, add_noise=False):
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))

        data = []
        label_vecs = []
        filenames = []
        pick_deterministically = (mode != 'training')

        for i in xrange(offset, offset + sample_count):
            # Pick which sample to use.
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                sample_index = np.random.randint(len(candidates))
            sample = candidates[sample_index]
            filename = sample['file']

            in_vec, label_vec = self.data[filename]
            if not pick_deterministically and add_noise:
                mu, std = 0., 0.001
                noise = np.random.normal(mu, std, len(in_vec))
                in_vec_ = in_vec + noise
                in_vec = np.array(_partial_normalize(in_vec_))

            filenames.append(filename)
            data.append(in_vec)
            label_vecs.append(label_vec)

        return filenames, np.array(data), np.array(label_vecs)
