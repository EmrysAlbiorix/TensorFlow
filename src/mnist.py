# To block some of the annoying red text messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

# Load the dataset
mnist = tf.keras.datasets.mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

# Normalize pixel values to 0-1 range
X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

tf.random.set_seed(42)
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(shape=[28, 28]))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

# Summary of the model
model.summary()

# Better optimizer, momentum based with steps and learning rate scheduling
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True
)
optimize = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model
model.compile(loss="sparse_categorical_crossentropy",
              #optimizer="sgd",
              optimizer=optimize,
              metrics=["accuracy"])

# Train the model (decreased epocks from 30 to 15 to cut training time)
history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                    validation_data=(X_valid, y_valid))

# Evaluate on test set
print("Test Set")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

#print(history.params)