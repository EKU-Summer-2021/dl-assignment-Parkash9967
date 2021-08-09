"""
Unit test module for Keras_model
"""
import unittest
import tensorflow as tf
import pandas as pd
from src.keras_model import KerasModel


class MyTestCase(unittest.TestCase):
    """
    Unit test class for Keras_model
    """

    def setUp(self):
        data = pd.read_csv(
            'https://raw.githubusercontent.com/Parkash9967/Test/08b6957aa8c693b60a66d741f2e1704ae73e536b/student-mat'
            '.csv')
        self.data = KerasModel(
            data[['age', 'Medu', 'Fedu', 'famrel', 'famrel', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G2',
                  'G1']],
            data['G3'])

    def test_grade_prediction_using_keras(self):
        """
           here we check if it is an instance of Keras Library
        """
        get_results = self.data.grade_prediction_using_keras()
        self.assertIsInstance(get_results, tf.keras.callbacks.History)
