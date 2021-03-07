from data.make_sims import read_json, write_json


if __name__ == '__main__':
    with open('valid_zhuyin.txt') as f:
        valids = [l for l in f.read().split('\n') if len(l) > 1 or (len(l) == 1 and ord(l) > 0x3119)]

    categories = read_json('category.json')
    mmap = read_json('map.json')
    sims = read_json('sim.json')

    psims = sims['prefix']
    ssims = sims['suffix']

    pron_sims = {z: {} for z in valids}
    zhuyins = list(sorted(valids))

    for i, z1 in enumerate(zhuyins):
        pron_sims[z1][z1] = 1.0
        for z2 in zhuyins[i + 1:]:
            p1, s1 = mmap[z1]
            p2, s2 = mmap[z2]

            pa, pb = list(sorted([p1, p2]))
            sa, sb = list(sorted([s1, s2]))

            ssim = 0.0 if sb not in ssims[sa] else ssims[sa][sb]

            try:
                psim = 0.0 if pb not in psims[pa] else psims[pa][pb]
            except:
                psim = 0.0

            sim = round(0.2 * psim + 0.8 * ssim, 2)

            if sim >= 0.75:
                pron_sims[z1][z2] = sim
                #pron_sims[z2][z1] = sim

    write_json(pron_sims, 'pron_sims.json')

    prefixes = list(sorted(categories['prefix']['category'].keys()))
    pre_idx = {key: i for i, key in enumerate(prefixes)}
    suffixes = list(sorted(categories['suffix']['category'].keys()))
    suf_idx = {key: i for i, key in enumerate(suffixes)}
