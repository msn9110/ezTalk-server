from subprocess import call
import get_current_state

from config import get_pid, read_test_log


def stop_exec_train():
    base_cmd = 'kill -9 '
    is_finish, state, _ = get_current_state.get_state()
    print()
    if is_finish:
        print('no training process')
        return
    else:
        to_stop = input('confirm to stop [y]/n: ')
        if to_stop == 'y':
            if state == 'training':
                pid = get_pid('worker')
                cmd = base_cmd + pid
                call(cmd, shell=True)
            elif state == 'testing':
                for i in range(4):
                    finished, content = read_test_log(i)
                    if not finished:
                        pid = content.split('\n')[0]
                        cmd = base_cmd + pid
                        call(cmd, shell=True)

            pid = get_pid('executor')
            cmd = base_cmd + pid
            call(cmd, shell=True)

            print('stopped')


if __name__ == '__main__':
    stop_exec_train()
