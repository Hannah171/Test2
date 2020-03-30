import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn import linear_model


data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1","G2","G3","studytime","failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict],1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

print("Co: \n", + linear.coef_)
print('Interpret: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print('we predict the grade is: ' + str(predictions[x]) + '\n', 'we have student information: ' + str(x_test[x]) + '\n', 'The true grade is: ' + str(y_test[x]) + '\n')
