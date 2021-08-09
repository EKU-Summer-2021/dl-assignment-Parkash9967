import numpy as np
from src import Polynomial
from src.keras_model import KerasModel
import pandas as pd

if __name__ == '__main__':
    coeffs = np.array([1, 0, 0])
    polynom = Polynomial(coeffs)
    print(polynom.evaluate(3))
    print(polynom.roots())
data = pd.read_csv(
    'https://raw.githubusercontent.com/Parkash9967/Test/08b6957aa8c693b60a66d741f2e1704ae73e536b/student-mat'
    '.csv')
data.drop(["school", "address", "sex", "famsize", "Pstatus", "Mjob", "Fjob",
           "reason", "guardian",
           "schoolsup", "famsup", "paid", "activities",
           "nursery", "higher", "internet", "romantic"],
          axis=1)
k = KerasModel(
    data[['age', 'Medu', 'Fedu', 'famrel', 'famrel', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G2', 'G1']],
    data['G3'])

print(k.grade_prediction_using_keras())
# print(k.save_results())
