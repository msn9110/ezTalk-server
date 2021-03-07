from matplotlib.font_manager import FontProperties
myfont = FontProperties(fname=r'/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
import re, os, argparse, glob
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

args = None
parts = ['', '聲母', '介音', '韻母']


def get_info(path):
    with open(path) as f:
        content = f.read()
    records = [re.sub(r'\s', '', rec) for rec in re.split(r'\n', content)\
               if rec.startswith('Partial')]
    records = [rec.split(':') for rec in records]
    records = [(k.strip('Partial(').strip(')'), (int(v.split('/')[0]), int(v.split('/')[1])))
               for k, v in records]
    records = list(sorted(records, key=lambda it: it[0]))
    return path, records


def draw_info(info):
    partial_files = []
    path, records = info

    partial_files.append(path)
    dst_dir, name = os.path.split(path)
    _part = int(os.path.split(dst_dir)[1][1])
    name = name.rstrip('.txt')
    new_name = re.sub('^result', 'partial', name) + '.txt'
    partial_files.append(os.path.join(dst_dir, new_name))

    labels = []
    values = []
    with open(os.path.join(dst_dir, new_name), 'w') as log_f:
        num_of_test = 0
        num_of_correct = 0
        for k, v in records:
            labels.append(k)
            num_of_test += v[1]
            num_of_correct += v[0]
            acc_percentage = int(round(v[0] / v[1], 2) * 100)
            values.append(acc_percentage)
            log_line = '{0:3s} : {1:d} / {2:d}, percentage -> {3:3d}%\n'.format(
                k, v[0], v[1], acc_percentage
            )
            log_f.write(log_line)
        total_percentage = round(num_of_correct / num_of_test * 100, 2)
        log_line = 'Total : {0:d} / {1:d}, percentage -> {2:.2f}%'.format(
            num_of_correct, num_of_test, total_percentage
        )
        log_f.write(log_line)

    top_num = int(name.split('result')[1][1])
    step = 50
    num_batch = int(len(labels) // step)
    num_batch = num_batch if len(labels) % step == 0 else num_batch + 1
    for i in range(0, len(labels), step):
        end = -1 if i + step > len(labels) else i + step
        num = int(i // step) + 1

        lx = labels[i:end]
        x = range(len(lx))
        y = values[i:end]

        fig_name = name + '-' + str(num) + '.png'
        fig_path = os.path.join(dst_dir, fig_name)
        title = '{0:s} top {1:d} : ({2:d} / {3:d})'.format(parts[_part],
                                                           top_num, num, num_batch)

        fig = plt.figure()

        plt.title(title, fontproperties=myfont)
        plt.ylim([0, 100])
        plt.ylabel('Accuracy(%)')
        plt.xlabel('label')
        plt.xticks(x, lx, fontproperties=myfont, rotation='vertical')
        plt.grid()
        plt.bar(x, y, edgecolor='black', color='g', linewidth=1.2)

        fig.set_size_inches(12, 10)
        fig.savefig(fig_path)
        plt.close(fig)

        partial_files.append(fig_path)
    return partial_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='', help='''\
                        specific a folder that contains result*.txt''')
    args, _ = parser.parse_known_args()
    if len(args.path) > 0 and os.path.isdir(args.path):
        from multiprocessing import Pool

        all_pic = glob.glob(os.path.join(args.path, 'result*.png'))
        for pic in all_pic:
            os.remove(pic)
        files = [f_path for f_path in glob.glob(os.path.join(args.path, 'result*.txt'))
                 if re.match('result_[1-5]', os.path.split(f_path)[1]) is not None]
        pool = Pool()
        all_info = pool.map(get_info, files)
        pool.close()
        pool.join()

        pool = Pool()
        all_partial_files = pool.map(draw_info, all_info)
        pool.close()
        pool.join()

        os.chdir(args.path)

        import shutil
        base_name = ''
        for a in re.split('/', args.path)[-3:]:
            base_name += a
        base_name += '_results'
        result_dir = os.path.join(args.path, base_name)
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir)

        for part in all_partial_files:
            for f in part:
                dst = os.path.join(result_dir, os.path.split(f)[1])
                shutil.copy(f, dst)

        shutil.make_archive(base_name, 'zip', '.', base_name)
        shutil.rmtree(result_dir)
    else:
        print('Please ENTER a valid path')
        exit(-1)