import os, argparse, shutil
from math import ceil
from config import restful_settings
import requests, json
import multiprocessing as mp

from matplotlib.font_manager import FontProperties
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

myfont = FontProperties(fname=r'/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')

e_parts = ['Syllable Comparison', 'Consonant Comparison',
           'Sound Comparison', 'Vowel Comparison']
parts = ['音節比較', '聲母比較', '介音比較', '韻母比較']

_markers = ['o', 'X', 's', 'd']
# custom colors
colors = ['#e5a1bd', '#c9a1e5',
          '#02c874', '#5dade2']


def worker(root, d):
    r = requests.get('http://' + restful_settings['ip'] + ':'
                     + restful_settings['service_port']
                     + '/results/{0:s}/'.format(args.user) + root)
    d[str(root)] = json.loads(r.text)


def draw(cmp_num, m, accuracies, name):
    if not 1 <= cmp_num:
        return
    tops = [1, 2, 3, 4, 5]
    top_num = len(tops)
    formatter = '{0:.1f}%'
    w = 1.0
    d = 1.0
    d /= 2

    gap_scale = cmp_num + 1
    x = [_ * gap_scale for _ in range(top_num)]
    start_offset_scale = \
        - ((cmp_num - 1) % 2 + ceil(cmp_num / 2 - 1) * 2)
    b_x = [[_ + (start_offset_scale + 2 * k) * d
            for _ in x]
           for k in range(cmp_num)]

    y = [0.0] * len(tops)  # empty data
    ticks = ['top ' + str(i) for i in tops]
    if not m:
        ticks = ['top {0}'.format(2 * i - 1)
                 for i in tops]
    fig = plt.figure()
    plt.title('{0:s}'.format(e_parts[m]),
              fontproperties=myfont, fontsize=16)

    # accuracy text in middle of bar
    for k in range(top_num):
        for idx in range(cmp_num):
            _x, acc = b_x[idx][k], accuracies[idx][k]
            _y = acc / 2
            text = '100%' if int(acc) == 100 \
                else formatter.format(acc)
            plt.text(_x, _y, text,
                     verticalalignment="center", horizontalalignment="center",
                     color='black')

    ax = plt.gca()

    plt.xticks(x, ticks)
    # let ticks in the middle
    ax.bar(x, y, width=0.0, align='center', color='white')

    bars = []
    # draw bars
    for idx in range(cmp_num):
        bar = ax.bar(b_x[idx], accuracies[idx], width=w,
                     facecolor=colors[idx], edgecolor='white',
                     align='center')
        bars.append(bar)
    lgd = plt.legend(bars, name, loc='upper center',
                     bbox_to_anchor=(1.145 - 0.0125 * (cmp_num - 1), 1))

    plt.ylim([0, 100])
    plt.ylabel('Accuracy(%)')
    ax.yaxis.grid()

    f_w = 7 + 3 * (cmp_num - 1)
    f_h = 5 + 0 * (cmp_num - 1)
    fig.set_size_inches(f_w, f_h)

    fig_path = '{0:s}/'.format(fig_dir)

    fig_path += parts[m] + '.png'
    fig.savefig(fig_path,
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    plt.close(fig)


def draw_cdfs(cdfs, names):
    if cdfs:
        tops = list(range(len(cdfs[0])))
        x = tops[::2]
        labels = map(str, x)
        cdfs = [[round(_, 1) for _ in cdf]
                for cdf in cdfs]
        fig = plt.figure()

        ax = plt.gca()

        plt.title('{0:s}'.format('CDF'),
                  fontproperties=myfont, fontsize=16)
        plt.ylim([0, 100])
        plt.ylabel('Percentage(%)', fontsize=14)
        plt.xlabel('Tops', fontsize=14)
        ax.yaxis.grid()
        ax.xaxis.grid()
        plt.xticks(x, labels, fontsize=12)

        # draw plots
        for idx in range(cmp_num):
            ax.plot(tops, cdfs[idx],
                    color=colors[idx], marker=markers[idx], lw=2.0, label=names[idx])
        lgd = ax.legend(loc='upper center',
                        bbox_to_anchor=(1.15, 1))

        f_w = 12
        f_h = 8
        fig.set_size_inches(f_w, f_h)

        fig_path = '{0:s}/'.format(fig_dir)
        fig_path += 'cdf.png'
        fig.savefig(fig_path,
                    bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
        plt.close(fig)


def draw_syllables(labels, rankings, names):
    if rankings:
        from math import ceil

        upper = 50
        y = list(range(0, upper + 1, 5))
        y[0] = 1
        yls = ['top {0:d}'.format(_) for _ in y]
        batch = 40
        num_subplots = int(ceil(len(labels) / batch))
        fig, axes = plt.subplots(num_subplots, 1)
        axes[0].set_title('Each Syllable 95% Comparison', fontsize=30)
        for n in range(num_subplots):
            ls = labels[n * batch:(n + 1) * batch]
            x = list(range(len(ls)))
            ax = axes[n]

            ax.set_ylim([1, upper])

            ax.grid()
            ax.set_yticks(y, minor=False)
            ax.set_yticklabels(yls, fontproperties=myfont,
                               fontsize=20,
                               fontdict=None, minor=False, )
            ax.set_xticks(x, minor=False)
            ax.set_xticklabels(ls, fontproperties=myfont,
                               fontsize=20,
                               fontdict=None, minor=False, )

            # draw plots
            for idx in range(cmp_num):
                r = rankings[idx][n * batch:(n + 1) * batch]
                r = [min(_, upper) for _ in r]
                ax.plot(x, r,
                        color=colors[idx], marker=markers[idx], lw=3.0,
                        label=names[idx])
            ax.legend(loc='upper center',
                      bbox_to_anchor=(1.08, 1))

        f_w = 20
        f_h = num_subplots * 10 + 2
        fig.set_size_inches(f_w, f_h)

        fig_path = '{0:s}/'.format(fig_dir)
        fig_path += '95cmp.png'
        fig.savefig(fig_path,
                    bbox_inches='tight')
        plt.close(fig)


# root contain part info
def draw_comparison(index, root, d):
    cmp_num = len(root)
    name = []
    data = []
    accuracies = []
    mode = int(index)
    for i in range(cmp_num):
        accuracies.append([])
        name.append(root[i].split("/")[0])
        data.append(d[root[i]])
    get_m_d = []

    mstr = "_" + str(mode)
    for k in range(cmp_num):
        get_m_d.append(next(d for i, d in
                            enumerate(data[k]["test_results"])
                            if mstr in d))

    if not mode:
        cdfs = []
        keys = set()
        my_rankings = []
        for k in range(cmp_num):
            ranking = get_m_d[k][mstr]['ranking']
            cdf = ranking['total'][2]['cdf']
            cdfs.append(cdf[:51])
            my_ranking = ranking['partial']
            my_ranking = dict([(it[0], it[3]) for it in my_ranking])
            if not keys:
                keys = set(my_ranking.keys())
            else:
                keys.intersection_update(my_ranking.keys())
            my_rankings.append(my_ranking)
        keys = list(sorted(keys))
        my_rankings = [[my_ranking[k] for k in keys]
                       for my_ranking in my_rankings]
        keys = ['\n'.join(k) for k in keys]
        draw_syllables(keys, my_rankings, name)

        cdf_bins = max([len(cdf) for cdf in cdfs])
        cdfs = [cdf + [cdf[-1]] * (cdf_bins - len(cdf))
                for cdf in cdfs]
        draw_cdfs(cdfs, name)
    end, step = (6, 1) if mode else (10, 2)
    try:
        for i in range(1, end, step):
            for k in range(cmp_num):
                accuracies[k].append(float(get_m_d[k][mstr]["total"]["accuracy"]["top " + str(i)].split("%")[0]))
            # draw
        draw(cmp_num, mode, accuracies, name)
    except KeyError:
        pass



def draw_task(args):
    mode, url, shared_d = args
    draw_comparison(mode, url, shared_d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--names', default='20190924-0.9-idx-err,', help='''\
                        e.g.: 20180913,''')
    parser.add_argument('-u', '--user', default='')
    args, _ = parser.parse_known_args()

    names = [_ for _ in args.names.split(',') if _]

    fig_dir = 'static/' + args.user + '/comparison'
    if os.path.exists(fig_dir):
        shutil.rmtree(fig_dir)
    os.makedirs(fig_dir)

    manager = mp.Manager()
    shared_dict = manager.dict()
    jobs = []
    root = []
    for name in names:
        r_str = name
        root.append(r_str)
        p = mp.Process(target=worker, args=(r_str, shared_dict))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    cmp_num = len(root)

    # add new colors
    delta = len(colors) - cmp_num
    from numpy.random import randint

    while delta < 0:
        color_nums = randint(100, 200, 3)
        c_str = '#'
        for c_num in list(color_nums):
            c_str += '{0:02x}'.format(c_num)
        if c_str not in colors:
            colors.append(c_str)
            delta += 1

    idxes = randint(0, len(_markers), cmp_num)
    markers = [_markers[i] for i in list(idxes)]

    para = [(str(i), root, shared_dict) for i in range(4)]
    pool = mp.Pool()
    pool.map(draw_task, para)
    pool.close()
    pool.join()
