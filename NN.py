import math
import random

class NeuralNetworkError(Exception):
  pass

class Neuron:
  @staticmethod
  def af_sigmoid(shift = 0):
    return lambda x: 1 / (1 + math.exp(-1. * (x - shift)))

  @staticmethod
  def af_threshold(a = 0):
    return lambda x: 1 if x >= a else 0

  def act_weight(self, y):
    t = self.activation_data[0]
    if t == 'sigmoid':
      return y * (1 - y)
    else:
      return 1

  def __init__(self, weights, activation=None):
    activation = activation or ('sigmoid', 0)
    self.weights = weights
    self.activation_data = activation
    act_type, act_param = activation or ('sigmoid', 0)
    if act_type == 'sigmoid':
      self.act_fn = Neuron.af_sigmoid(act_param)
    elif act_type == 'threshold':
      self.act_fn = Neuron.af_threshold(act_param)

  def __repr__(self):
    act_type, act_param = self.activation_data
    return "<Neuron activation=%s(%s), weights=(%s)>" % (act_type, act_param, ", ".join("%0.2f" % a for a in self.weights))

  def apply(self, inputs):
    return self.act_fn(sum(map(lambda x: x[0] * x[1], zip(self.weights, inputs))))

class NeuralNetwork:
  def __init__(self, num_inputs=2):
    self.num_inputs = num_inputs
    self.layers = []
    self.final = False

  def __repr__(self):
    return "<NeuralNetwork: %s>" % ", ".join(map(lambda l: "(%s)" % ", ".join(map(lambda n: "(%s)" % ", ".join("%0.2f" % a for a in n.weights), l)), self.layers))

  def add_layer(self, num_neurons, activation=None):
    if self.final:
      raise NeuralNetworkError('Cannot add new layer; output layer has already been added.')
    length = self.num_inputs if len(self.layers) is 0 else len(self.layers[-1])
    layer = []
    for i in range(num_neurons):
      n = Neuron([random.random() for i in range(length)], activation)
      layer.append(n)
    self.layers.append(layer)

  def add_output_layer(self, num_neurons=1, activation=None):
    self.add_layer(num_neurons, activation)
    self.final = True

  def forward(self, inputs):
    outputs = [inputs]
    for layer in self.layers:
      outputs.append(tuple(map(lambda n: n.apply(outputs[-1]), layer)))
    return outputs

  def backward(self, outputs, goal, rate):
    # calculate errors
    delta = map(lambda x: (x[0] - x[1]), zip(goal, outputs[-1]))
    deltas = [[] for i in range(len(self.layers))]
    deltas[-1] = delta
    for i in range(len(self.layers)-2, -1, -1):
      cur, next = self.layers[i], self.layers[i + 1]
      d = []
      for j in range(len(cur)):
        d.append(reduce(lambda a, p: p[0] * p[1] + a, zip(map(lambda n: n.weights[j], next), deltas[i + 1]), 0))
      deltas[i] = d
    # change weights
    paired = map(lambda p: zip(*p), zip(self.layers, deltas))
    for i in range(len(paired)):
      layer = paired[i]
      for j in range(len(layer)):
        neuron, delta = layer[j]
        neuron.weights = map(sum, zip(neuron.weights, map(lambda x: x * neuron.act_weight(outputs[i + 1][j]) * rate * delta, outputs[i])))

  def apply(self, inputs):
    return self.forward(inputs)[-1]

  def run(self, training_data, runs=1, learning_rate=0.5):
    for i in range(runs):
      for x, y in training_data:
        y = y if type(y) == type(tuple()) else (y,)
        # feed forward
        outputs = self.forward(x)
        self.backward(outputs, y, learning_rate)

nn = NeuralNetwork(64)
nn.add_layer(8)
nn.add_layer(8)
nn.add_output_layer(1)

data = [
        # 4
        ((0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0, 0,0,1,0,0,0,0,0, 0,0,1,0,1,0,0,0, 0,0,1,1,1,1,0,0, 0,0,0,0,1,0,0,0, 0,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,0), 0.5),
        ((0,0,0,0,0,0,0,0, 0,0,1,0,1,0,0,0, 0,0,1,0,1,0,0,0, 0,0,1,0,1,0,0,0, 0,0,1,1,1,1,0,0, 0,0,0,0,1,0,0,0, 0,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,0), 0.5),
        ((0,0,0,0,0,0,0,0, 0,0,0,0,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,1,0,1,0,0,0, 0,0,1,1,1,1,0,0, 0,0,0,0,1,0,0,0, 0,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,0), 0.5),
        # 1
        ((0,0,0,0,0,0,0,0, 0,0,0,0,1,0,0,0, 0,0,0,0,1,0,0,0, 0,0,0,0,1,0,0,0, 0,0,0,0,1,0,0,0, 0,0,0,0,1,0,0,0, 0,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,0), 0.2),
        ((0,0,0,0,0,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0,0,0,0,0,0), 0.2),
        # 2
        ((0,0,0,0,0,0,0,0, 0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, 0,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0, 0,0,0,1,0,0,0,0, 0,1,1,1,1,1,1,0, 0,0,0,0,0,0,0,0), 0.3),
        # 3
        ((0,0,0,0,0,0,0,0, 0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, 0,0,0,0,0,0,1,0, 0,0,1,1,1,1,0,0, 0,0,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,0,1,1,1,1,0,0), 0.4),
        ((0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, 0,0,0,0,0,0,1,0, 0,0,1,1,1,1,0,0, 0,0,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,0,1,1,1,1,0,0, 0,0,0,0,0,0,0,0), 0.4),
        # 8
        ((0,0,0,0,0,0,0,0, 0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,0,1,1,1,1,0,0), 0.9),
        ((0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,0,1,1,1,1,0,0, 0,0,0,0,0,0,0,0), 0.9),
        # garbage
        ((1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1), 0),
        ((0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0), 0)
       ]

nn.run(data, learning_rate=0.75, runs=500)

def print_img(image):
  result = []
  for i in range(0, 64, 8):
    result.append("".join(map(lambda n: "X" if n is 1 else " ", image[i:i+8])))
  return "\n".join(result)

for image, value in data:
  print print_img(image),
  print int(round(10 * nn.apply(image)[0])) - 1