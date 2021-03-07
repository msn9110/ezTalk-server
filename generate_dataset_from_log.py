import re, os, argparse

from config import get_settings


def generate_dataset(log_path, mode, name):
    if log_path.endswith('.json'):
        import json
        with open(log_path) as log_f:
            data = json.load(log_f)
            all_path = []
            for _, fs in data[mode].items():
                all_path.extend(fs)
    else:
        with open(log_path) as log_f:
            content = log_f.read()
        content = re.sub(r'[\[\]]', '', content)
        lines = re.split(r'\n+', content)
        all_path = [line for line in lines if re.match(r'/.*\.wav', line) is not None]

    clip_name = '_clip_' + str(settings['clip_duration'])
    source_dir = settings['voice_data_path']['source'] + '/' + clip_name
    stn_dir = settings['voice_data_path']['sentence'] + '/' + clip_name
    files = [re.split('/', p)[-2:] for p in all_path]
    dst_root_dir = VOICE_DATA_DIR + '/' + mode + '_' + name
    total = 0
    import shutil
    if os.path.exists(dst_root_dir):
        shutil.rmtree(dst_root_dir)
    for label, f in files:
        dst_dir = dst_root_dir + '/' + label
        dst = dst_dir + '/' + f

        src = source_dir + '/' + label + '/' + f
        if os.path.exists(src):
            print(src)
            os.makedirs(dst_dir, exist_ok=True)
            os.symlink(src, dst)
            total += 1
            continue

        src = stn_dir + '/' + label + '/' + f
        if os.path.exists(src):
            print(src)
            os.makedirs(dst_dir, exist_ok=True)
            os.symlink(src, dst)
            total += 1
    print(total, '/', len(all_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lp', '--log_path', default='')
    parser.add_argument('-m', '--mode', default='test', help='''\
                        test, train''')
    parser.add_argument('-n', '--name', default='(name)')
    parser.add_argument('-u', '--user', default='')
    args, _ = parser.parse_known_args()
    if len(args.log_path) == 0:
        print('enter log path')
        exit(-1)
    *_, (*_, VOICE_DATA_DIR), _, settings = get_settings(args.user)

    generate_dataset(args.log_path, args.mode, args.name)
