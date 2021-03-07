import os
import json
import datetime as dt
from math import log2 as log
from math import fabs
from functools import wraps


def time_used(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = dt.datetime.now()
        print("function", func.__name__, 'starts to work at {}'.format(start.strftime("%H:%m:%S")))
        results = func(*args, **kwargs)
        end = dt.datetime.now()
        delta_time = end - start
        print("function", func.__name__, 'finishes at {}'.format(end.strftime("%H:%m:%S")))
        print("function", func.__name__, 'spends', str(dt.timedelta(seconds=delta_time.total_seconds())))
        return results
    return wrapper


def custom_sigmoid_base(prob):
    return pow(2.0, (log(prob / (1 - prob) / prob)))


def get_my_sigmoid_base(v, remained_at_v):
    return pow(2.0, -log(1 / remained_at_v - 1) / v)


# - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  TOOL: normalize sum to one
# - - - - - - - - - - - - - - - - - - - - - - - - - - -
def normalize(list_):
    results = []
    for syllables in list_:
        if len(syllables):
            total = sum([prob for _, prob in syllables])
            results.append([(syllable, prob / total)
                            for syllable, prob in syllables])
        else:
            results.append([])
    return results


# - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  TOOL: convert log value to probability
# - - - - - - - - - - - - - - - - - - - - - - - - - - -
def log_to_prob(lists_):
    possible_lists = []
    if type(lists_[0]) == list:
        epsilon = 1e-40
        temp = [{k: fabs(log(max(v, epsilon))) for k, v in a}
                for a in lists_]
        lists_ = temp

    for temp_dict in lists_:
        sum_log = sum(list(temp_dict.values()))
        count_log = len(temp_dict.items())
        result = {k: round((sum_log - v) / (sum_log * (count_log - 1)), 6)
                  if count_log > 1 else 1.000
                  for k, v in temp_dict.items()}
        result = list(sorted(result.items(), key=lambda it: it[1],
                             reverse=True))
        possible_lists.append(result)

    return possible_lists


# pre-process probability function for constructed mode
def process_probability(lst, to_process=False):
    return log_to_prob(lst) if to_process else lst


def read_json(file_path):
    with open(file_path, 'r') as f:
        d = json.load(f)

    return d


def write_json(d, file_path):
    parent = os.path.split(file_path)[0]

    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(d, f, ensure_ascii=False, indent=2, sort_keys=True)


def clean_up_settings(settings):

    import re
    default_path_prefix = settings['__default_path_prefix']

    def delete_private_attribute(d):
        for k in list(d.keys()):
            if k.startswith('__'):
                del d[k]
                continue

            if isinstance(d[k], dict):
                d[k] = delete_private_attribute(d[k])

        return d

    def lstrip(d, prefix):
        for k in d.keys():
            if isinstance(d[k], dict):
                d[k] = lstrip(d[k], prefix)
            elif isinstance(d[k], str):
                d[k] = re.sub('^' + prefix, '', d[k])

        return d

    settings = delete_private_attribute(settings)

    for k, v in default_path_prefix.items():
        settings[k] = lstrip(settings[k], v + '/')

    return settings


def write_settings(settings, path):
    settings = clean_up_settings(settings)
    write_json(settings, path)
