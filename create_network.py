from NeuralNetwork import NeuralNetwork as NN
from keras.datasets import mnist

# Import MNIST dataset
(train_data, train_labels), (test_data, test_labels) = mnist.load_data(path="mnist.npz")
img_dim = train_data[0].shape

network = NN(img_dim, 2, 10, [20, 10])

output = network.runNetwork(train_data[0])
print(output)