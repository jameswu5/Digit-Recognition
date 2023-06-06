from network import NeuralNetwork
from game import Canvas, Game
from training import TrainingData
from keras.datasets import mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data()

def simulate():
    C = Canvas()
    NN = NeuralNetwork([784, 16, 16, 10], 'digits_network_info.txt')
    G = Game(C, NN)
    G.simulate()

def train_network(batch_size, iterations):
    training_set = [TrainingData(train_x[i], train_y[i]) for i in range(len(train_x))]
    NN = NeuralNetwork([784, 16, 16, 10], 'digits_network_info.txt')
    NN.train(training_set, batch_size, iterations)
    NN.write_to_file('digits_network_info.txt')

def test_network():
    test_set = [TrainingData(test_x[i], test_y[i]) for i in range(len(test_x))]
    NN = NeuralNetwork([784, 16, 16, 10], 'digits_network_info.txt')
    NN.test(test_set)

if __name__ == "__main__":
    #train_network(100, 5)
    #test_network()
    simulate()