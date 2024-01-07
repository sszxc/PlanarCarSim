import functools
import time


def cprint(str, color):
    '''construnct a string with color print, by Xuechao
    color: 'black', 'red', 'green', 'yellow', 'blue', 'purple', 'cyan', 'white'
    '''
    color_dict = {'black':30, 'red':31, 'green':32, 'yellow':33, 'blue':34, 'purple':35, 'cyan':36, 'white':37}
    print(f'\033[1;{color_dict[color]};40m{str}\033[0m')

def timer(func):
    '''
    用函数装饰器实现函数计时
    '''
    @functools.wraps(func)
    def wrapper(*args, **kw):
        starttime = time.time()
        f = func(*args, **kw)
        endtime = time.time()
        print('Called %s(), ' % func.__name__, end='')
        print(f'Elapsed time {endtime - starttime:.2f} s.')
        return f
    return wrapper