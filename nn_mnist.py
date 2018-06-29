import gzip
import _pickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = _pickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set



# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

#plt.imshow(train_x[12].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print (train_y[57])


# TODO: the neural net!!
train_y=one_hot(train_y,10)
valid_y=one_hot(valid_y,10)
test_y=one_hot(test_y,10)

imagenes = tf.placeholder("float", [None, 784])  # imagen
etiquetas = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 15)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(15)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(15, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(imagenes, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(etiquetas - y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print ("----------------------")
print ("   Start training...  ")
print ("----------------------")

batch_size = 20

error = 999999
errors = []
epoch = 1

while True:
    for jj in range(int(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={imagenes: batch_xs, etiquetas: batch_ys})

    print ("Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={imagenes: batch_xs, etiquetas: batch_ys}))
    result = sess.run(y, feed_dict={imagenes: batch_xs})
    errorPrev = error
    error = sess.run(loss, feed_dict={imagenes: valid_x, etiquetas: valid_y})
    errors.append(error)
    print ("Validacion : Epoch #:", epoch, "Error: ", error)

    if(abs(error-errorPrev)/errorPrev) < 0.001 or error > errorPrev:
        break
    epoch = epoch + 1
aciertos = 0.
test_result = sess.run(y, feed_dict={imagenes: test_x})
for operacion, dato in zip(test_result, test_y):
    if (np.argmax(operacion) == np.argmax(dato)):
        aciertos = aciertos + 1

tasaAciertos= (aciertos/len(test_result)*100)

print ("La tasa de aciertos obtenida con el conjunto de test es : ", tasaAciertos, "%")

plt.plot(errors)
plt.title("Evolucion del error de entrenamiento")
plt.xlabel("Epocas")
plt.ylabel("Error")
plt.savefig("graficaErrores.png")
plt.show()