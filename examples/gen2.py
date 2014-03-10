from GA import *
from math import sqrt

def split(c):
    return [i+1 for i in range(len(c)) if c[i] == 0], [i+1 for i in range(len(c)) if c[i] == 1]

def p(c):
    return str(tuple(split(c)))

def calc(c):
    sumPile, prodPile = split(c)
    def product(l):
        r = 1
        for i in l:
            r *= i
        return r
    return (sum(sumPile), product(prodPile))

def error(target):
    def errorFn(c):
        return sqrt(sum((a-b)**2 for a,b in zip(target, calc(c))))
    return errorFn

cards = GAModel(15)
cards.setPrint(p)
cards.setError(error((3+4+6+7+13+14+15-5, 1*2*5*8*9*10*11*12)))

g = GA(cards, 100, 0.75, 0.51)
g.run(2000)