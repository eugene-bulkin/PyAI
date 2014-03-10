import random

class GAModel:
    # errorFn takes a chromosome and returns something on [0, infinity)
    def __init__(self, chromosomeSize, errorFn=None, printFn=None):
        self.CS = chromosomeSize
        self.error = errorFn
        self.printFn = printFn

    def setPrint(self, printFn):
        self.printFn = printFn

    def setError(self, errorFn):
        self.error = errorFn

    def printC(self, chromosome):
        if self.printFn:
            return self.printFn(chromosome)
        else:
            return self.__str__()

    def fitness(self, c):
        if self.error is None:
            raise Exception("No error function to calculate fitness with")
        return 1 / (1 + self.error(c))

    def generate(self):
        return tuple(random.randint(0,1) for i in range(self.CS))

class GA:
    def __init__(self, model, popSize, crossOver, mutation, topRank=5):
        self.N = popSize
        self.CR = crossOver
        self.MR = mutation
        self.TR = topRank

        self.model = model

    def get_results(self, population, fitnesses):
        res_p, res_f = [], []
        for c, f in zip(population, fitnesses):
            if c not in res_p:
                res_p.append(c)
                res_f.append(f)
        top = sorted(zip(res_p, res_f), key = lambda t: (-t[1],len(t[0])))[:self.TR]
        return "\n".join("%s; e = %.1f, f = %.3e" % (self.model.printC(c), self.model.error(c), f) for c, f in top)

    def select(self, population, fitnesses, n = 1):
        p, f = population[:], fitnesses[:]
        result = []
        for i in range(n):
            total = sum(f)
            selection = random.random() * total
            running, j = 0, 0
            while selection > running:
                running += f[j]
                j += 1
            result.append(p[j-1])
            del p[j - 1]
            del f[j - 1]
        return result

    def crossover(self, c1, c2):
        crossBit = random.randint(0, self.model.CS)
        return (c1[:crossBit] + c2[crossBit:], c1[crossBit:] + c2[:crossBit])

    def mutate(self, c):
        new = []
        for i in c:
            if random.random() < self.MR:
                new.append(1 - i)
            else:
                new.append(i)
        return tuple(new)

    def run(self, iterations=100):
        population, fitnesses = [], []
        cur_iter = 0
        correct = False

        # generate initial population
        for i in range(self.N):
            c = self.model.generate()
            population.append(c)
            fitnesses.append(self.model.fitness(c))

        while cur_iter < iterations:
            new_pop, new_fit = [], []
            while len(new_pop) < self.N:
                chr1, chr2 = self.select(population, fitnesses, 2)
                if random.random() < self.CR:
                    chr1, chr2 = self.crossover(chr1, chr2)
                chr1, chr2 = self.mutate(chr1), self.mutate(chr2)
                f1, f2 = self.model.fitness(chr1), self.model.fitness(chr2)
                if f1 > f2:
                    new_pop.append(chr1)
                    new_fit.append(f1)
                else:
                    new_pop.append(chr2)
                    new_fit.append(f2)
            population, fitnesses = new_pop, new_fit
            cur_iter += 1
            if 1 in fitnesses: # found a solution
                form = (self.model.printC(population[fitnesses.index(1)]), cur_iter, self.TR, self.get_results(population, fitnesses))
                print "CORRECT: %s (iteration #%i)\nTop %i results:\n%s" % form
                correct = True
                break
            if cur_iter % (iterations / 10) == 0:
                print "Iteration #%i" % cur_iter
        if not correct:
            print "Found no solution. Top %i results:\n%s" % (self.TR, self.get_results(population, fitnesses))

    def __call__(self, iterations=100):
        self.run(iterations)

