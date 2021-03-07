from config import read_log, read_test_log


def get_state():
    args = read_log('state')
    state = args[0]
    print(' ' * 80, end='\r')
    if state == 'finish':
        msg = state
        print(state)
        return True, state, msg
    if state == 'training':
        mode, times, num_of_models = [int(_) for _ in args[1:]]
        step = int(read_log('step')[0])
        if mode:
            base = (mode - 1) / num_of_models
            offset = step / (num_of_models * times)
            percentage = round(100 * (base + offset), 2)
        else:
            percentage = round(100 * step / times, 2)
        msg = 'model {0} {1} total finish : {2:.2f} %'.format(mode, state, percentage)
    elif state == 'testing':

        msg = ''
        for m in range(4):
            repeat = True
            while repeat:
                finished, msg_ = read_test_log(m)
                repeat = not finished
                if not finished:
                    try:
                        _, total, step = msg_.split('\n')
                        msg_ = '_{0:d} test:{1:s}/{2:s}'.format(m, step, total)
                        repeat = False
                    except:
                        repeat = True
                        continue
                msg += msg_ + ' ' * 1
    else:
        msg = '{0}'.format(state)

    print(msg, end='\r')
    return False, state, msg


if __name__ == '__main__':
    import time
    msg = ''
    try:
        while True:
            is_finished, _, msg = get_state()
            if is_finished:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print(' ' * 50, end='\r')
        print(msg, end='\r')
        print()
