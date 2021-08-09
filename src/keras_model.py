"""
module for predicting the grades using keras library
"""
from datetime import datetime
import os
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


class KerasModel:
    """
    class KerasModel with constructor
    """

    def __str__(self):
        return self.__class__.__name__

    def __init__(self, data_x, data_y):
        self.model = keras.Sequential(
            [
                layers.Dense(2, activation="relu", name="layer1"),
                layers.Dense(3, activation="relu", name="layer2"),
                layers.Dense(4, name="layer3"),
            ]
        )
        self.data_x = data_x
        self.data_y = data_y
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data_x, self.data_y, test_size=0.20
                                                                                , random_state=42)

    def grade_prediction_using_keras(self):
        """
            function of keras module to predict the grades
        """
        x_test = self.x_test.astype('float32')

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(21, activation=tf.nn.softmax))

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        history = self.model.fit(self.x_train, self.y_train, epochs=100)

        score = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        # result = pd.DataFrame({'Test loss': score[0], 'Test accuracy': score[1]
        #                        })
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        y_predict = self.model.predict(x_test)
        print(y_predict)
        self.model.summary()
        return history

    def save_results(self):
        """
          function of keras module to save results into separate directory
        """
        history = self.grade_prediction_using_keras()
        path_dir = os.getcwd()
        directory = 'Results_GradePrediction'
        directory_dt = 'GP'
        path_results = path_dir + r'/Results_GradePrediction'
        path_dr = path_dir + r'/Results_GradePrediction/GP'
        if not os.path.isdir(path_results):
            if not os.path.exists(path_results):
                os.makedirs(os.path.join(path_dir, directory), exist_ok=True)
            if not os.path.exists(path_dr):
                os.makedirs(os.path.join(path_results, directory_dt), exist_ok=True)
        file_name = 'Plot' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
        file_location = os.path.join(path_dr, file_name)
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.savefig(file_location + ".png")
        plt.show()
