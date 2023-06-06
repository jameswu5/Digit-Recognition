import numpy as np
import math
import random

LEARN_RATE = 1

class NeuralNetwork:
    def __init__(self, layer_sizes, import_file = None):
        self.layers = []
        self.number_of_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        if import_file is None:
            self.weights = self.initialise_weights()
            self.biases = self.initialise_biases()
        else:
            self.weights, self.biases = self.read_file(import_file)
        self.learn_rate = LEARN_RATE

    def initialise_weights(self):
        weights = []
        for i in range(self.number_of_layers - 1):
            initial_weight = np.random.normal(0, 0.5, [self.layer_sizes[i], self.layer_sizes[i+1]])
            weights.append(initial_weight)
        return weights
    
    def initialise_biases(self):
        biases = []
        for i in range(self.number_of_layers - 1):
            initial_bias = np.zeros(self.layer_sizes[i+1])
            biases.append(initial_bias)
        return biases

    def activation_function(self, value):
        return 1 / (1 + math.exp(-value))
    
    def one_forward_pass(self, input_layer, weight_matrix, bias_vector):
        output_layer = np.matmul(input_layer, weight_matrix) + bias_vector
        # Apply the activation function
        func = np.vectorize(self.activation_function)
        return func(output_layer)

    def forward_propogate(self, input_vector):
        self.layers = [] # Reset the layers
        func = np.vectorize(self.activation_function)
        current_vector = func(input_vector) # Normalise the input vector
        self.layers.append(current_vector)

        for i in range(self.number_of_layers - 1):
            weight_matrix = self.weights[i]
            bias_vector = self.biases[i]
            current_vector = self.one_forward_pass(current_vector, weight_matrix, bias_vector)
            self.layers.append(current_vector)
        
        return current_vector

    def calculate_node_values_output_layer(self, expected_output):
        """
        Mean Squared Error is our cost function so the derivative of (a-y)^2 = 2(a-y)
        Activation function is sigmoid so S'(x) = S(x) * (1 - S(x))
        """
        node_values = np.zeros(self.layer_sizes[-1])
        output_layer = self.layers[-1]
        for i in range(self.layer_sizes[-1]):
            cost_derivative = 2 * (output_layer[i] - expected_output[i])
            activation_derivative = output_layer[i] * (1 - output_layer[i])
            node_values[i] = cost_derivative * activation_derivative
        return node_values
    
    def calculate_node_values_hidden_layer(self, hidden_layer_nodes, node_values_of_higher_layer, weight_matrix):
        node_values = np.matmul(weight_matrix, np.transpose(node_values_of_higher_layer))
        for i in range(len(node_values)):
            layer_node = hidden_layer_nodes[i]
            activation_derivative_layer_node = layer_node * (1 - layer_node)
            node_values[i] = node_values[i] * activation_derivative_layer_node
        return node_values

    def generate_node_values(self, expected_output):
        node_values = []
        node_values_output_layer = self.calculate_node_values_output_layer(expected_output)
        node_values.append(node_values_output_layer)
        #Iterate through hidden layers
        for i in range(self.number_of_layers - 2):
            weight_matrix = self.weights[self.number_of_layers - 2 - i]
            # Check the hidden layer index in self.layers
            node_values_hidden_layer = self.calculate_node_values_hidden_layer(self.layers[self.number_of_layers-2-i], node_values[i], weight_matrix)
            node_values.append(node_values_hidden_layer)
        node_values.reverse()
        return node_values

    def back_propogate(self, expected_output):
        node_values = self.generate_node_values(expected_output)
        
        weight_derivatives = []
        bias_derivatives = []
        for layer_index in range(self.number_of_layers - 1):
            weight_derivative = np.zeros((self.layer_sizes[layer_index], self.layer_sizes[layer_index + 1]))
            for in_node_index in range(self.layer_sizes[layer_index]):
                for out_node_index in range(self.layer_sizes[layer_index + 1]):
                    weight_derivative[in_node_index][out_node_index] = self.layers[layer_index][in_node_index] * node_values[layer_index][out_node_index]
            weight_derivatives.append(weight_derivative)

            bias_derivative = np.zeros(self.layer_sizes[layer_index + 1])
            for node_index in range(self.layer_sizes[layer_index + 1]):
                bias_derivative[node_index] = 1 * node_values[layer_index][node_index]
            bias_derivatives.append(bias_derivative)
        
        return weight_derivatives, bias_derivatives

    def gradient_descent(self, weight_gradients, bias_gradients):
        for i in range(self.number_of_layers - 1):
            self.weights[i] -= weight_gradients[i] * self.learn_rate
            self.biases[i] -= bias_gradients[i] * self.learn_rate

    def check_if_correct(self, output_vector, expected_output):
        return np.argmax(output_vector) == np.argmax(expected_output)

    def train_batch(self, training_batch):
        weight_gradients = []
        bias_gradients = []
        for i in range(self.number_of_layers - 1):
            matrix = np.zeros((self.layer_sizes[i], self.layer_sizes[i+1]))
            weight_gradients.append(matrix)

            array = np.zeros(self.layer_sizes[i+1])
            bias_gradients.append(array)

        average_cost = 0
        correct = 0
        
        for trainingdata in training_batch:
            input_vector, expected_output = trainingdata.input_vector, trainingdata.expected_output
            # Evaluating performance
            output_vector = self.forward_propogate(input_vector)
            correct += self.check_if_correct(output_vector, expected_output)
            difference = expected_output - output_vector
            average_cost += np.linalg.norm(difference) / len(training_batch)

            weight_derivatives, bias_derivatives = self.back_propogate(expected_output)
            for i in range(self.number_of_layers - 1):
                weight_gradients[i] += weight_derivatives[i] / len(training_batch)
                bias_gradients[i] += bias_derivatives[i] / len(training_batch)
        print(correct, average_cost)
        self.gradient_descent(weight_gradients, bias_gradients)

    def train(self, training_set, batch_size, iterations):
        count = 0
        for _ in range(iterations):
            random.shuffle(training_set)
            for i in range(len(training_set) // batch_size):
                count += 1
                print(count, end= " ")
                training_batch = training_set[i * batch_size: (i+1) * batch_size]
                self.train_batch(training_batch)

    def write_to_file(self, filename):
        f = open(filename, "w")
        f.write(" ".join(map(str, self.layer_sizes)) + "\n")
        for weight_matrix in self.weights:
            for row in weight_matrix:
                f.write(" ".join(map(str, row)) + "\n")
        for bias_vector in self.biases:
            f.write(" ".join(map(str, bias_vector)) + "\n")

    def read_file(self, filename):
        weights = []
        biases = []

        f = open(filename, "r")
        lines = list(f.readlines())
        layer_sizes = list(map(int, lines[0].split()))

        index = 1
        for layer_index in range(len(layer_sizes) - 1):
            weight_matrix = []
            for _ in range(layer_sizes[layer_index]):
                weight_matrix.append(list(map(float, lines[index].split())))
                index += 1
            weights.append(weight_matrix)
        for _ in range(len(layer_sizes) - 1):
            biases.append(list(map(float, lines[index].split())))
            index += 1

        return weights, biases

    def test(self, test_set):
        correct = 0
        count = 0
        for test_case in test_set:
            count += 1
            output_vector = self.forward_propogate(test_case.input_vector)
            if self.check_if_correct(output_vector, test_case.expected_output):
                correct += 1
            if count % 1000 == 0:
                print(str(correct), str(count), str(round(correct/count, 3)))