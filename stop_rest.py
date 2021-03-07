from subprocess import call
from config import get_pid

if __name__ == '__main__':
    cmd = 'kill -9 ' + get_pid('restful')
    call(cmd, shell=True)
