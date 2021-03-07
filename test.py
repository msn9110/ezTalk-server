import fcntl
from time import  sleep
new_entry = "foobar\n"
timeout = 60
count = 0
flag = False
while count < timeout:
    try:
        g = open('lock')
    except IOError:
        g = open('lock', 'a')
    try:
        fcntl.flock(g, fcntl.LOCK_EX|fcntl.LOCK_NB)
        #g.write(new_entry)
        _ = input('enter any :')
        flag = True
    except OSError:
        count += 1
        print('waiting...', end='\n')
        sleep(1)
    finally:
        fcntl.flock(g, fcntl.LOCK_UN)
        g.close()
        if flag:
            break

print('\ndone!')