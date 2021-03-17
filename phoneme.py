import re
from config import general_data_path
from utils import read_json

element = read_json(general_data_path['zmap'])
prefixes, suffixes = set(), set()

for v in element.values():
    prefixes.add(v[0])
    suffixes.add(v[1])

with open(general_data_path['selected']) as f:
    selected = f.read().split('\n')

# include optional label
_i_opt = 1
_o_opt = 1
_s_opt = [2, 2]
optionals = [['_unknown_', '_silence_'],
             ['_unknown_1', '_silence_1'],
             ['_unknown_2', '_silence_2'],
             ['_unknown_3', '_silence_3'],
             ['_unknown_4', '_silence_4'],
             ['_unknown_5', '_silence_5'],
             ['_unknown_6', '_silence_6']]
_c = 1
tops = optionals[_i_opt * _c][:_s_opt[0]] + \
       ['no_label1'] + [chr(i) for i in range(0x3105, 0x311a)]
_c += 1
mids = optionals[_i_opt * _c][:_s_opt[0]] + \
       ['no_label2'] + [chr(i) for i in range(0x3127, 0x312a)]
_c += 1
bots = optionals[_i_opt * _c][:_s_opt[0]] + \
       ['no_label3'] + [chr(i) for i in range(0x311a, 0x3127)]
_c += 1
prefs = optionals[_i_opt * _c][:_s_opt[0]] + list(sorted(prefixes))
_c += 1
suffs = optionals[_i_opt * _c][:_s_opt[0]] + list(sorted(suffixes))
_c += 1
sels = optionals[_i_opt * _c][:_s_opt[0]] + list(sorted(selected))
_c = 0
# input converter
indexes = [{k: i for i, k in enumerate(l)}
           for l in [tops, mids, bots, prefs, suffs, sels]]
rev_indexes = [{l[k]: k for k in l.keys()}
               for l in indexes]

with open(general_data_path['valid_zhuyin']) as f:
    valid_zhuyins = optionals[_o_opt * _c][:_s_opt[1]] + \
                    [z for z in sorted([_ for _ in f.read().split('\n') if _])]
output_labels = valid_zhuyins
# output converter
zindexes = {k: i for i, k in enumerate(output_labels)}
rev_zindexes = {zindexes[k]: k for k in zindexes.keys()}

indexes = [zindexes] + indexes
rev_indexes = [rev_zindexes] + rev_indexes

sim_tables = read_json(general_data_path['sim_tables'])


def get_syllable(wav_label, mode=''):
    m = {'top': '1', 'mid': '2', 'bot': '3'}
    suffix = m[mode] if mode in m else ''

    def get_label(s):
        return s if len(s) > 0 else 'no_label' + suffix

    if wav_label in ['_unknown_', '_silence_']:
        return wav_label

    if mode == 'top':
        syllable = re.sub('[^\u3105-\u3119]', '', wav_label)
    elif mode == 'bot':
        syllable = re.sub('[^\u311a-\u3126]', '', wav_label)
    elif mode == 'mid':
        syllable = re.sub('[^\u3127-\u3129]', '', wav_label)
    elif mode == 'pre':
        return element[wav_label][0]
    elif mode == 'suf':
        return element[wav_label][1]
    else:
        return wav_label

    return get_label(syllable)


mmap = {m: i for i, m in enumerate(['all', 'top', 'mid', 'bot', 'pre', 'suf', 'sel'])}
idx_mode = {i: m for m, i in mmap.items()}


zelements = {z: [get_syllable(z, m)
                 for m in ['top', 'mid', 'bot', 'pre', 'suf']]
             for z in valid_zhuyins}


if __name__ == '__main__':
    prons_sims = read_json(general_data_path['pron_sims'])
    sims = read_json(general_data_path['sims'])
    similarities = {
        'all': prons_sims,
        'pre': sims['prefix'],
        'suf': sims['suffix']
    }

    tables = {
        mode: {
            s: [0.0] * len(indexes[mmap[mode]])
            for s in indexes[mmap[mode]].keys()
        }
        for mode in ['all', 'pre', 'suf']
    }

    for mode in ['all', 'pre', 'suf']:
        positions = indexes[mmap[mode]]
        table = tables[mode]
        s = similarities[mode]
        for k in positions.keys():
            idx1 = positions[k]
            table[k][idx1] = 1.0
            if k in s:
                for k2 in s[k].keys():
                    idx2 = positions[k2]
                    v = s[k][k2]
                    table[k][idx2] = v
                    table[k2][idx1] = v

    from utils import write_json
    write_json(tables, 'data/sim_tables.json')
