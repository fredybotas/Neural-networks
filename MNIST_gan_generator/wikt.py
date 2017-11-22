import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_data_x = np.genfromtxt('MNIST_gan_generator/x_train.csv',delimiter=',')
temp = np.genfromtxt('MNIST_gan_generator/y_train.csv',delimiter=',')
train_data_y = []
for a in temp:
    if a == 1.:
        train_data_y.append([0., 1.])
    else:
        train_data_y.append([1., 0.])


validation_data_x = np.genfromtxt('../wikt/x_validation.csv',delimiter=',')
temp = np.genfromtxt('../wikt/y_validation.csv',delimiter=',')
validation_data_y = []
for a in temp:
    if a == 1.:
        validation_data_y.append([0., 1.])
    else:
        validation_data_y.append([1., 0.])

input_dim = len(train_data_x[0])
h1_dim = 128 #size of hidden layer
h2_dim = 128
output_dim = 2

learning_rate = 0.01
epoch_range = 50000
batch_size = 100


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# network initialization
x = tf.placeholder("float32", shape=[None, input_dim])
y = tf.placeholder("float32", shape=[None, output_dim])

N_W1 = tf.Variable(xavier_init([input_dim, h1_dim]))
N_B1 = tf.Variable(np.zeros(shape=[h1_dim]))
N_B1 = tf.cast(N_B1, "float32")

N_W2 = tf.Variable(xavier_init([h1_dim, h2_dim]))
N_B2 = tf.Variable(np.zeros(shape=[h2_dim]))
N_B2 = tf.cast(N_B2, "float32")

N_W3 = tf.Variable(xavier_init([h2_dim, output_dim]))
N_B3 = tf.Variable(np.zeros(shape=output_dim))
N_B3 = tf.cast(N_B3, "float32")

hidden_layer1 = tf.nn.sigmoid(tf.matmul(x, N_W1) + N_B1)
hidden_layer2 = tf.nn.sigmoid(tf.matmul(hidden_layer1, N_W2) + N_B2)
output_layer = tf.matmul(hidden_layer2, N_W3) + N_B3

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
prediction = tf.cast(prediction, "float32")
accuracy = tf.reduce_mean(prediction)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

counter = 0
for epoch in range(epoch_range):
    batch_x = train_data_x[counter: counter+batch_size]
    batch_y = train_data_y[counter: counter+batch_size]
    counter += batch_size
    if counter >= len(train_data_x):
        counter = 0
    if epoch % 100 == 0:
        training = accuracy.eval({x: batch_x, y:batch_y})
        print("Training accuracy " + str(training))

    sess.run([optimizer, loss], feed_dict={x:batch_x, y:batch_y})

print(accuracy.eval({x: validation_data_x, y:validation_data_y}))
