import numpy as np

class TrainingData:
    def __init__(self, image_matrix, label):
        self.label = label
        self.image_matrix = image_matrix
        self.input_vector = self.generate_input_vector(image_matrix)
        self.expected_output = self.generate_expected_vector(label)

    def generate_input_vector(self,image_matrix):
        return image_matrix.flatten()
    
    def generate_expected_vector(self,label):
        expected = np.zeros(10)
        expected[label] = 1
        return expected