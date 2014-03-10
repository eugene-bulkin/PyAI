from GA import *

genes = list('0123456789+-*/')
def enc(input):
    return genes.index(input)

def dec(output):
    if output >= len(genes):
        return None
    return genes[output]
def is_op(symbol):
    return symbol in list('+-*/')

def is_num(symbol):
    return symbol in list('0123456789')

# takes GAModel chromosome and cleans it up for this purpose
def clean(c):
    try:
        chromosome = [int(''.join(str(a) for a in c[max(0,i - 4):i]), 2) for i in range(4,len(c) + 1, 4)]
    except ValueError:
        print c
        exit()
    result = []
    i, t = 0, 0
    last_op = '+'
    while i < len(chromosome):
        symb = dec(chromosome[i])
        if symb is None:
            i += 1
            continue
        if t % 2 == 0 and not is_num(symb): # even indices must be numbers
            i += 1
            continue
        if t % 2 == 1 and not is_op(symb): # odd indices must be operators
            i += 1
            continue
        if t % 2 == 0:
            if last_op == '/' and symb == '0':
                i += 2
                t += 2
                continue
        else:
            last_op = symb
        result.append(chromosome[i])
        i += 1
        t += 1
    try:
        if is_op(dec(result[-1])): # last one can't be operator
            result = result[:-1]
    except IndexError:
        return []
    return result

def calc(c):
    chromosome = clean(c)
    result = 0
    i = 0
    t = 0 # type counter; must be number when even, operator when odd
    last_op = '+' # default to last operator being addition
    while i < len(chromosome):
        symbol = dec(chromosome[i])
        if t % 2 == 0: # just got a number
            number = float(symbol)
            if last_op == '+':
                result += number
            elif last_op == '-':
                result -= number
            elif last_op == '*':
                result *= number
            elif last_op == '/':
                result /= number
        else:
            last_op = symbol
        i += 1
        t += 1
    return result

def print_c(c):
    chromosome = clean(c)
    result = ""
    decoded = [dec(i) for i in chromosome]
    paren = False
    for symb in decoded:
        result += symb
        if paren:
            result = "(" + result + ") "
            paren = False
        else:
            result += " "
        if symb in list('+-'):
            paren = True
    while result[-1] == " ":
        result = result[:-1]
    return result

def error(target):
    def errorFn(c):
        return abs(target - calc(c))
    return errorFn

Operations = GAModel(35 * 4)
Operations.setPrint(print_c)

c = Operations.generate()

g = GA(Operations, 100, 0.75, 0.1)

target = 1
while target != 0:
    try:
        target = int(raw_input("target? "))
    except KeyboardInterrupt:
        break
    if target == 0:
        break;

    Operations.setError(error(target))

    g(1500)