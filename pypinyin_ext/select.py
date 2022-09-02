from pypinyin_ext.zhuyin import convert_to_zhuyin as pinyin

if __name__ == '__main__':
    path = '/home/hungshing/Data/Downloads/spoken.txt'
    with open(path) as f:
        lines = f.read().split('\n')
    d = {l: [z[0].strip('ˋ˙ˊˇ')
             for z in pinyin(l)]
         for l in lines}
    print(d)

    keys = {k for converted in d.values()
            for k in converted}
    print(len(keys))
    print(keys)

    with open('/home/hungshing/FastData/ezTalk/apps/data/valid_zhuyin.txt') as f:
        all_zhuyin = f.read().split('\n')
        if not all_zhuyin[-1]:
            all_zhuyin = all_zhuyin[:-1]

    mlist = {z: []
             for z in all_zhuyin}

    for z in all_zhuyin:
        for stn, zstn in d.items():
            if z in zstn:
                mlist[z].append([stn, ','.join(zstn)])

    from utils import write_json, read_json
    unuse = []
    for z in all_zhuyin:
        if not mlist[z]:
            del mlist[z]
            unuse.append(z)
    write_json(mlist, 'help_to_select.json')
    zc_d = read_json('/home/hungshing/FastData/ezTalk/apps/data/word_default_weights.json')
    ref_word = {z: zc_d[z] for z in unuse}
    write_json(ref_word, 'unused.json')
