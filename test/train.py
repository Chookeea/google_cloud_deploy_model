import os

'''
Default to 0, so all logs are shown. Set TF_CPP_MIN_LOG_LEVEL to 1 to filter out
INFO logs, 2 to additional filter our WARNING, 3 to additionally filter out ERROR.

You can adjust the verbosity by changing the value of TF_CPP_MIN_LOG_LEVEL:

0 = all messages are logged (default behavior)
1 = INFO message are not printed
2 = INFO and WARNING message are not printed
3 = INFO, WARNING, and ERROR messages are not printed

'''

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)


# normalize: 0,255 -> 0,1
x_train, x_test = x_train / 255.0, x_test / 255.0

# model 
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10),
])


# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
'''
Adam is an optimization algorithm that can be used instead of the classcical stochastic
gradient descent procedure to update network weights iterative based in training data.

The authors describe Adam as combining the advantages of two other extensions of stochastic gradient descent. Specifically:

Adaptive Gradient Algorithm (AdaGrad) that maintains a per-parameter learning rate that improves performance on problems with sparse gradients (e.g. natural language and computer vision problems).
Root Mean Square Propagation (RMSProp) that also maintains per-parameter learning rates that are adapted based on the average of recent magnitudes of the gradients for the weight (e.g. how quickly it is changing). This means the algorithm does well on online and non-stationary problems (e.g. noisy).
'''

optim = keras.optimizers.Adam(lr=0.001)
metrics = ['accuracy']

model.compile(loss=loss, optimizer=optim, metrics=metrics)

# training
batch_size = 64
epochs = 5

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

'''
The Hierachical Data Format version 5 (HDF5), is an open source file format that support large, complex, heterogeneous data.
HDF5 uses a "file directory" structure that allows you to organize data within the file in many different structured ways, as you might do with files on your computer.

'''

model.save("nn.h5") # .h5 = HDF5

# evaluate
print("Evaluate both models:")
model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)

new_model = keras.models.load_model("nn.h5")
new_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)

