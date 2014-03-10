from GA import *
from math import sqrt

def split(c):
    return [i+1 for i in range(len(c)) if c[i] == 0], [i+1 for i in range(len(c)) if c[i] == 1]

def p(c):
    return tuple((2*n - 1) * a[i] for n, i in zip(c, range(l)))

target = [1, -6, 5, -3, -2, 4, -4, -5, 8, 3, 7, 2, -5, -9, 4]
a = tuple(abs(a) for a in target)
l = len(target)

def errorFn(c):
    s =  sum((2*n - 1) * a[i] for n, i in zip(c, range(l)))
    return 1./(1. + abs(s))

cards = GAModel(l)
cards.setPrint(p)
cards.setError(errorFn)

g = GA(cards, 200, 0.85, 0.21)
g.run(1500)