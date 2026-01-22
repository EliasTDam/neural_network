import numpy as np

class NeuralNetwork:

    # Initialize network with input dimensionality (as a list), number of layers and number of neurons in each layer,
    # or path to saved network
    def __init__(self, input_dim=None, n_hidden_layers=None, output_dim=None, n_neurons=None, path_to_network=None):

        # List containing all the weights between layers
        self.network = []
        self.weights = []
        self.biases = []

        # Learning rate
        self.a = 0.1

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

            weights = np.random.rand(n_neurons[i], output_size_last)
            bias = np.random.rand(n_neurons[i])
            output_size_last = n_neurons[i]

            self.weights.append(weights)
            self.biases.append(bias)

            layer = [weights, bias]

            self.network.append(layer)

        # Create final connection to output layer
        last_weights = np.random.rand(output_dim, output_size_last)
        last_bias = np.random.rand(output_dim)
        self.weights.append(last_weights)
        self.biases.append(last_bias)
        self.network.append([last_weights, last_bias])

    def trainNetwork(self, training_data, labels, epochs=1000, batch_size=100):


        for epoch in range(epochs):
            #print(f"PRE SHUFFLE Training data: {training_data[:, 10:15, 10:15]} \n Labels: {labels}")
            perm = np.random.permutation(training_data.shape[0])
            #print(f"PERM: {perm}")
            training_data = training_data[perm]
            labels = labels[perm]
            #print(f"POST SHUFFLE Training data: {training_data[:, 10:15, 10:15]} \n Labels: {labels}")
            for batch_itr in range(int(np.ceil(training_data.shape[0]/batch_size))):
                if (batch_itr+1)*batch_size > training_data.shape[0]:
                    batch_data = training_data[batch_itr*batch_size:]
                    batch_labels = labels[batch_itr*batch_size:]
                else:
                    batch_data = training_data[batch_itr*batch_size:(batch_itr+1)*batch_size]
                    batch_labels = labels[batch_itr*batch_size:(batch_itr+1)*batch_size]
                output = self.trainSingleBatch(batch_data, batch_labels)
            print(f"Epoch {epoch} out of {epochs} completed")
            print(f"Accuracy: {self.testNetwork(training_data, labels)}")

    def trainSingleBatch(self, data, labels):

        nabla_biases = [np.zeros(b.shape) for b in self.biases]
        nabla_weights = [np.zeros(w.shape) for w in self.weights]

        # Loop over all training examples
        for instance in range(data.shape[0]):

            d_nabla_bias, d_nabla_weight = self.backpropogation(data[instance], labels[instance])
            nabla_weights = [n + (d / data.shape[0]) for n, d in zip(nabla_weights, d_nabla_weight)]
            nabla_biases = [n + (d / data.shape[0]) for n, d in zip(nabla_biases, d_nabla_bias)]

        # Apply the change
        for layer in range(len(self.weights)):
            self.weights[layer] = self.weights[layer] + self.a * nabla_weights[layer]
            self.biases[layer] = self.biases[layer] + self.a * nabla_biases[layer]



    def backpropogation(self, input_data, target_value):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feed forward through the network, and save the activations in each layers
        activation = np.array(input_data).flatten()
        activations = [np.array(input_data).flatten()]  # List to store all the activations, layer by layer
        zs = [] # List to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = w @ activation + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        delta = self.calculateCostDerivative(target_value, activations[-1]) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, len(self.weights)):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l].transpose())
        return (nabla_b, nabla_w)

        """
        # Feed forward through the network, and save the activations in each layers
        activation = np.array(input_data).flatten()
        activations = [np.array(input_data).flatten()]  # List to store all the activations, layer by layer
        zs = [] # List to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.calculateCostDerivative(target_value, activations[-1]) * \
                self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, len(self.weights)):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)
    """

    def runNetwork(self, input_data, return_all=False):

        # Feed forward through the network, and save the activations in each layers
        activation = np.array(input_data).flatten()
        for b, w in zip(self.biases, self.weights):
            z = w @ activation + b
            activation = self.sigmoid(z)

        output = activation

        if return_all:
            return output
        else:
            return np.argmax(output)

    def testNetwork(self, test_data, labels):
        correct = 0
        for i in range(test_data.shape[0]):
            prediction = self.runNetwork(test_data[i])
            if prediction == labels[i]:
                correct += 1
        return correct / test_data.shape[0]


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def calculateCost(self, target_value, value_array):
        target_array = np.zeros(value_array.shape)
        target_array[target_value] = 1
        cost = (target_array - value_array)**2

        return cost

    def calculateCostDerivative(self, target_value, value_array):
        target_array = np.zeros(value_array.shape)
        target_array[target_value] = 1

        return (value_array - target_array)