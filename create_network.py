from NeuralNetwork import NeuralNetwork as NN
from keras.datasets import mnist

# Import MNIST dataset
(train_data, train_labels), (test_data, test_labels) = mnist.load_data(path="mnist.npz")
img_dim = train_data[0].shape
print(f"Train data: {train_data.shape}")

network = NN(img_dim, 2, 10, [20, 10])

success_rate = network.testNetwork(test_data, test_labels)
print(f"Initial success rate: {success_rate}")

network.trainNetwork(train_data, train_labels, epochs=50)

success_rate = network.testNetwork(test_data, test_labels)
print(f"Final success rate: {success_rate}")