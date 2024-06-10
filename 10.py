import numpy as np
import tensorflow as tf

class SimpleRBM:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = tf.Variable(tf.random.normal([input_size, output_size], 0.01))
        self.h_bias = tf.Variable(tf.zeros([output_size]))
        self.v_bias = tf.Variable(tf.zeros([input_size]))

    def sample(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))

    def step(self, v):
        h_probs = tf.nn.sigmoid(tf.matmul(v, self.W) + self.h_bias)
        h_sample = self.sample(h_probs)
        v_probs = tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(self.W)) + self.v_bias)
        v_sample = self.sample(v_probs)
        return h_sample, v_sample

    def train(self, data, epochs=1000, lr=0.1):
        for epoch in range(epochs):
            for v in data:
                v = np.reshape(v, (1, -1))
                h_sample, v_sample = self.step(v)
                pos_grad = tf.matmul(tf.transpose(v), h_sample)
                neg_grad = tf.matmul(tf.transpose(v_sample), self.step(v_sample)[0])
                self.W.assign_add(lr * (pos_grad - neg_grad))
                self.v_bias.assign_add(lr * tf.reduce_mean(v - v_sample, axis=0))
                self.h_bias.assign_add(lr * tf.reduce_mean(h_sample - self.step(v_sample)[0], axis=0))

    def transform(self, data):
        data = np.reshape(data, (1, -1))
        h_probs = tf.nn.sigmoid(tf.matmul(data, self.W) + self.h_bias)
        return h_probs

# Parameters
input_size = 6
hidden_size_1 = 3
hidden_size_2 = 2
epochs = 1000
learning_rate = 0.1

# Sample data
data = np.array([[1, 1, 1, 0, 0, 0],
                 [1, 0, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [0, 0, 1, 1, 1, 0],
                 [0, 0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 1, 0]], dtype=np.float32)

# Train first RBM
rbm1 = SimpleRBM(input_size, hidden_size_1)
rbm1.train(data, epochs, learning_rate)
h1 = np.array([rbm1.transform(v) for v in data])

# Train second RBM
rbm2 = SimpleRBM(hidden_size_1, hidden_size_2)
rbm2.train(h1, epochs, learning_rate)
h2 = np.array([rbm2.transform(h) for h in h1])

# Print results
print("Original Data:\n", data)
print("Features from RBM1:\n", h1)
print("Features from RBM2:\n", h2)