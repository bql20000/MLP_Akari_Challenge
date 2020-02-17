import numpy as np
import MyMLP
import pandas as pd
import Visualization
from sklearn.model_selection import train_test_split

# todo: handle input
df = pd.read_csv('data/train.csv')
labels = np.asarray(df['label'])
images = df.drop(columns="label").values

# todo: feature engineering
# rescaling
images = images / 255.0
# use 10000 samples and split data (8/2)
X_train, X_test, y_train, y_test = train_test_split(images[0:5000], labels[0:5000], test_size=0.2, random_state=0)


# todo: implement MyMLP
mlp = MyMLP.MyMLP((1200,), C=10, learning_rate=0.1, max_iter=10, momentum=0.9)
mlp.print()
mlp.fit(X_train, y_train)

# todo: testing
pred = mlp.predict(X_test)
score = 100 * np.mean(pred == y_test)
print("Test accuracy : %.2f %%" % score)


# Please UNCOMMENT parts you want to experiment

# todo: EXPERIMENT with different hidden dimension
# ----------------------------------------------------------------------------------------------------------------------
"""
list_units = [100, 200, 500, 1000, 1500, 2000]
list_score_train = []
list_score_test = []
list_loss = []
for units in list_units:
    mlp = MyMLP.MyMLP((int(units),), C=10, learning_rate=0.1, max_iter=200, momentum=0.9)
    mlp.fit(X_train, y_train)
    list_score_test.append(mlp.predict_score(X_test, y_test))
    list_score_train.append(mlp.predict_score(X_train, y_train))
    list_loss.append(mlp.loss())
Visualization.visualize(list_units, [list_loss, list_score_train, list_score_test], ['Loss', 'Train accuracy', 'Test accuracy'], xlabel='Number of hidden units', title="2-layer perceptron")
"""



# todo: EXPERIMENT with different epochs
# ----------------------------------------------------------------------------------------------------------------------
"""
list_epochs = [50, 100, 200, 500, 2000]
list_loss = []
list_score = []
for epoch in list_epochs:
    mlp = MyMLP.MyMLP((100,), C=10, learning_rate=0.1, max_iter=epoch, momentum=0.9)
    mlp.fit(X_train, y_train)
    list_score.append(mlp.predict_score(X_test, y_test))
    list_loss.append(mlp.loss())
Visualization.visualize(list_epochs, [list_loss, list_score], ['Loss', 'Test accuracy'], xlabel='Number of epochs', title="2-layer perceptron")
"""


# todo: EXPERIMENT with different input scaling
# ----------------------------------------------------------------------------------------------------------------------
"""
mlp_scaled = MyMLP.MyMLP((100,), C=10, learning_rate=0.1, max_iter=100, momentum=0.9)
mlp_scaled.fit(X_train, y_train)
mlp_not_scaled = MyMLP.MyMLP((100,), C=10, learning_rate=0.1, max_iter=100, momentum=0.9)
mlp_not_scaled.fit(X_train, y_train)
score_scaled = mlp_scaled.predict_score(X_test, y_test)
score_not_scaled = mlp_not_scaled.predict_score(X_test, y_test)
print("Training accuracy of scaled input and non-scaled input: (%.2f %%, %.2f %%)" % (score_scaled, score_not_scaled))
"""





# todo: EXPERIMENT with different learning rate
# ----------------------------------------------------------------------------------------------------------------------
"""
list_learning_rate = [0.01, 0.05, 0.1, 0.5, 1]
list_loss = []
list_score = []
for lr in list_learning_rate:
    mlp = MyMLP.MyMLP((100,), C=10, learning_rate=lr, max_iter=5000, momentum=0.9)
    mlp.fit(X_train, y_train)
    list_score.append(mlp.predict_score(X_test, y_test))
    list_loss.append(mlp.loss())
Visualization.visualize(list_learning_rate, [list_loss], ['Loss'], xlabel='Number of epochs', title="2-layer perceptron")
"""



# todo: EXPERIMENT with different momentum
# ----------------------------------------------------------------------------------------------------------------------
"""
list_momentum = np.linspace(0.0, 0.9, 10)
list_score = []
list_loss = []
for units in list_momentum:
    mlp = MyMLP.MyMLP((100,), C=10, learning_rate=0.1, max_iter=200, momentum=0.9)
    mlp.fit(X_train, y_train)
    list_score.append(mlp.predict_score(X_test, y_test))
    list_loss.append(mlp.loss())
Visualization.visualize(list_momentum, [list_loss, list_score], ['Loss', 'Test accuracy'], xlabel='Momentum', title="2-layer perceptron")
"""






