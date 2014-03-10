import random

BIAS = 1
RESPONSE = 0.5

class Neuron:

    def __init__(self, n):
        self.weights = []
        self.num_inputs = n

        for i in range(0, n + 1):
            # append random weight between -1 and 1
            self.weights.append(2 * random.random() - 1)

class NeuronLayer:
    def __init__(self, num, inputsper):
        self.num_neurons = 0
        self.neurons = []

class NeuralNet:

    def __init__(self):
        self.num_inputs = 0
        self.num_outputs = 0
        self.num_hidden_layers = 0
        self.neurons_per_hlayer = 0
        self.layers = []

    def createNet(self):
        pass

    def getWeights(self):
        pass

    def getNumWeights(self):
        pass

    # replaces the weights with new ones
    def putWeights(self, weights):
        pass

    # calculates the outputs from a set of inputs
    def update(inputs):
        outputs = []
        cWeight = 0

        if len(inputs) != self.num_inputs:
            return outputs

        for i in range(0, self.num_hidden_layers + 1):
            if i > 0:
                inputs = outputs

            outputs = []
            cWeight = 0
            # for each neuron sum (inputs * corresponding weights). Throw
            # the total at our sigmoid function to get the output
            for j in range(0, self.layers[i].num_neurons):
                netInput = 0
                numInputs = self.layers[i].neurons[j].num_inputs

                # for each weight
                for k in range(0, numInputs - 1):
                    cWeight += 1
                    netInput += self.layers[i].neurons[j].weights[k] * inputs[cWeight]

                netInput += self.layers[i].neurons[j].weights[numInputs - 1] * BIAS

                outputs.push(self.sigmoid(netInput, RESPONSE))

                cWeight = 0
        return outputs

    def sigmoid(activation, response):
        pass