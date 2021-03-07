import os
import json
import time
from math import log2 as log
from math import fabs
from functools import wraps


def time_used(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        s = time.time()
        print(f"function {func.__name__} start to work")
        results = func(*args, **kwargs)
        print(f"function {func.__name__} takes {time.time() - s} sec")
        return results
    return wrapper


def custom_sigmoid_base(prob):
    return pow(2.0, (log(prob / (1 - prob) / prob)))


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


def write_json(d, file_path, sort_keys=True):
    parent = os.path.split(file_path)[0]

    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(d, f, ensure_ascii=False, indent=2, sort_keys=sort_keys)
