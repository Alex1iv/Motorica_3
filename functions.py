# Файл для хранения функций для ноутбука третьего этапа соревнований от Моторики

def fib(n):    # write Fibonacci series up to n
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
    print()

def privet(name):   # print 'privet' to a given name
    return f'privet {name}'
