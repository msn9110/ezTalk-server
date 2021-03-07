import os
import json
import numpy as np


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


def div(a, b, default=1.0):
    return default if a + b == 0 else a / b


def sim(m_fixes, m_idx, table, is_prefix=True, **kwargs):
    m_sims = {_: {} for _ in m_fixes}

    c = 0.2 if is_prefix else 0.33
    d = cls[is_prefix]

    for i, a in enumerate(m_fixes):
        m_sims[a][a] = 1.0

        for b in m_fixes[i + 1:]:
            vec_a = table[m_idx[a]]
            vec_b = table[m_idx[b]]

            s = sim = div(sum(vec_a * vec_b), max(sum(vec_a), sum(vec_b)), **kwargs)

            if a[-1] == b[-1]:
                s = sim = min(1.0, c + sim)

            if is_prefix:
                sim *= float(d[a] == d[b])

                abend = ''.join(sorted([a[-1], b[-1]]))

                if abend == 'ㄧㄩ' and sim == 0.0 and len(a) == len(b)\
                        and abs(d[a[0]] - d[b[0]]) < 1:
                    sim = max(0.1, s)

            else:
                if a[0] == b[0]:
                    sim *= 0.25

                abend = ''.join(sorted([a[0], b[0]]))

                if len(b) > len(a) and a != b[-1] or \
                    (len(a) == len(b) == 2 and a[0] != b[0]) or sim < 0.2:
                    sim = 0.0

                if abend == 'ㄧㄩ' and a[-1] == b[-1] or \
                    a + b == 'ㄧㄩ':
                    sim = max(0.7, s)

            if sim > 0.0:
                m_sims[a][b] = round(sim, 2)
            #m_sims[b][a] = sim
    return m_sims


if __name__ == '__main__':
    with open('valid_zhuyin.txt') as f:
        valids = set([l for l in f.read().split('\n') if l])

    categories = read_json('category2.json')

    prefixes = list(sorted(categories['prefix']['category'].keys()))
    pre_idx = {key: i for i, key in enumerate(prefixes)}
    suffixes = list(sorted(categories['suffix']['category'].keys()))
    suf_idx = {key: i for i, key in enumerate(suffixes)}

    with open('pre.txt') as f:
        pre = {z: i for i, g in enumerate(f.read().split('-'))
               for z in g.split(',')}

    with open('suf.txt') as f:
        suf = {z: i for i, g in enumerate(f.read().split('-'))
               for z in g.split(',')}

    cls = {True: pre, False: suf}

    pre_suf = np.zeros([len(prefixes), len(suffixes)])
    suf_pre = np.zeros([len(prefixes), len(suffixes)]).transpose()

    # fill valid combination with 1
    for pf in prefixes:
        p_id = pre_idx[pf]

        for sf in suffixes:
            s_id = suf_idx[sf]

            if len(pf) > len(sf):
                continue

            if pf in 'ㄧㄨㄩ' and pf != sf[0]:
                continue

            if pf[-1] == sf[0]:
                sf = sf[1:]

            pron = pf + sf

            if pron in valids:
                pre_suf[p_id][s_id] = 1

    for sf in suffixes:
        s_id = suf_idx[sf]

        for pf in prefixes:
            p_id = pre_idx[pf]

            if len(pf) < len(sf):
                continue

            if sf in 'ㄧㄨㄩ' and sf != pf[-1]:
                continue

            if pf[-1] == sf[0]:
                pf = pf[:-1]

            pron = pf + sf

            if pron in valids:
                suf_pre[s_id][p_id] = 1

    prefix_sims = sim(prefixes, pre_idx, pre_suf, is_prefix=True)
    suffix_sims = sim(suffixes, suf_idx, suf_pre, is_prefix=False, default=0.0)

    simularities = {'prefix': prefix_sims,
                    'suffix': suffix_sims}

    write_json(simularities, 'simularity.json')
