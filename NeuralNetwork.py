import numpy as np

class NeuralNetwork:

    # Initialize network with input dimensionality (as a list), number of layers and number of neurons in each layer,
    # or path to saved network
    def __init__(self, input_dim=None, n_hidden_layers=None, output_dim=None, n_neurons=None, path_to_network=None):

        # List containing all the weights between layers
        self.network = []

        # Create network if no existing network is provided
        if path_to_network is None:
            self.createNetwork(input_dim, n_hidden_layers, output_dim, n_neurons)

    # Create network from scratch, and initialize with random values
    def createNetwork(self, input_dim, n_layers, output_dim, n_neurons):

        input_length = np.prod(input_dim)

        if np.shape(n_neurons)[0] != n_layers:
            raise ValueError(f"Number of entries in list of neurons per layers must match number of layers, but the"
                             f" current number of layers is {n_layers} and the number of entries in the list "
                             f"is {np.shape(n_neurons)[0]}")

        output_size_last = input_length
        for i in range(n_layers):

            print(f"Creating layer with shape {n_neurons[i], output_size_last}")

            weights = np.random.rand(n_neurons[i], output_size_last)
            bias = np.random.rand(n_neurons[i])
            output_size_last = n_neurons[i]

            layer = [weights, bias]

            self.network.append(layer)

        # Create final connection to output layer
        self.network.append(np.random.rand(output_dim, output_size_last))

    def runNetwork(self, input_data, return_all=False):

        # Flatten input data
        data = np.array(input_data).flatten()

        # Push the input through the network
        for layer in self.network:
            data = layer[0] @ data + layer[1]

        output = data

        if return_all:
            return output
        else:
            return np.argmax(output)