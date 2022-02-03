import numpy as np

def activation_function(element):
    # ReLu activation function
    return max(0, element)

activation_function_vectorized = np.vectorize(activation_function)

def compute(input_units, weights1, weights2):
    direction = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT"
    }

    input_units = np.insert(input_units, 0, 1, axis=0)
    input_units = np.reshape(input_units, (25, 1))
    temp = np.matmul(weights1, input_units)
    temp = activation_function_vectorized(temp)
    temp = np.insert(temp, 0, 1, axis=0)
    output = np.matmul(weights2, temp)
    output = activation_function_vectorized(output)

    index = np.argmax(output)
    return direction[index]
