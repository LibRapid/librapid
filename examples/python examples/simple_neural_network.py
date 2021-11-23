import random  # For the training process

import librapid

"""
In this example program, we create a *very* simple
neural network using LibRapid.

The neural network trains on the XOR binary operation,
which takes two bits and returns true if ONLY one bit
is enabled. The truth table is below:

A | B | A XOR B
---------------
0 | 0 |    0
0 | 1 |    1
1 | 0 |    1
1 | 1 |    0
"""

# First, define the inputs and outputs

# Note, the inputs and outputs are reshaped
# to be a list of column vectors (i.e. n by 1)
# in order to fit into the neural network. In
# later versions, this will be done automatically

# The inputs (A and B) to the XOR operation
inputs = librapid.from_data(
    [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
).reshaped(4, 2, 1)

# The outputs for the XOR operation (A XOR B)
outputs = librapid.from_data(
    [0, 1, 1, 0]
).reshaped(4, 1, 1)

# Create the config for the neural network.
# Many options and parameters can be used, which
# can be found in the documentation

# Create a neural network with 2 inputs, one hidden
# layer with 3 nodes, and one output node, with
# a learning rate of 0.2
config = {
    "input": 2,
    "hidden": [4],
    "output": 1,
    "learning rate": 0.01,
    "activation": "sigmoid",
    "optimizer": "sgd_momentum"
}

# Create the neural network
my_network = librapid.network(config)

# Compile the neural network
my_network.compile()

# Train the neural network
print("Training neural network")
epochs = 50000 * 4  # 5000 times through the data
for i in range(epochs):
    index = random.randint(0, 3)
    my_network.backpropagate(inputs[index], outputs[index])

print("Evaluating neural network output")
for i in range(4):
    output = float(my_network.forward(inputs[i])[0][0])
    print("Input:", inputs[i].transposed(), " => ", librapid.round(output, 3))
