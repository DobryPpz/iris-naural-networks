from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import keras.metrics
from keras.utils import to_categorical


#0 = iris-setosa
#1 = iris-versicolor
#2 = iris-virginica

dataset = loadtxt("iris.csv",delimiter=",")
X = dataset[:,0:4]
y = dataset[:,4]

model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',keras.metrics.Precision(), keras.metrics.Recall()])
y = to_categorical(y)

model.fit(X, y, epochs=1000, batch_size=10)

# _, accuracy = model.evaluate(X, y)
# print("==========================================")
# print("accuracy =",accuracy)
# print("==========================================")


