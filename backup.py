import waveTools as wt
import re
import os
import shutil


def make_file_list(path, is_sentence=False, output_name='remote_files.txt',
                   write=False):
    print(path)
    info = wt.get_all_waves(path, write_info=False, is_sentence=is_sentence)
    all_files = []
    all_source_files = []
    for label, files in info:
        all_files.extend([re.sub(os.path.abspath(path) + '/', '', f) for f in files])
        all_source_files.extend(files)

    if write:
        with open(path + '/' + output_name, 'w') as f:
            for a in all_files:
                f.write(a + '\n')

    mappings = {k: src for k, src in zip(all_files, all_source_files)}
    return all_files, mappings


def make_copies(path, compare_f_path, is_sentence=False):
    with open(compare_f_path) as c_f:
        compare_files = [f for f in re.split(r'\n', c_f.read()) if len(f) > 0]
        all_files, mappings = make_file_list(path, is_sentence)
        diff_files = [f for f in all_files if f not in compare_files]
        if os.path.exists(path + '/bkup'):
            shutil.rmtree(path + '/bkup')
        for f in diff_files:
            label = f.split('/')[0]
            dst_dir = path + '/bkup/' + label
            os.makedirs(dst_dir, exist_ok=True)
            dst = path + '/bkup/' + f
            shutil.copy(mappings[f], dst)


if __name__ == '__main__':
    from config import settings
    src_dir = settings['source']
    stn_dir = settings['sentence']
    remote_file = src_dir + '/remote_files.txt'
    make_copies(src_dir, remote_file)
    remote_file = stn_dir + '/remote_files.txt'
    make_copies(stn_dir, remote_file, True)
