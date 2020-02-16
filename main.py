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
# split data (8/2)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

"""
# todo: implement
mlp = MyMLP.MyMLP((100,), C=10, learning_rate=0.2, max_iter=100, momentum=0.9)
mlp.print()
mlp.fit(X_train, y_train)

# todo: testing
pred = mlp.predict(X_test)
score = 100 * np.mean(pred == y_test)
print("Accuracy score: %.2f %%" % score)
"""

# ----------------------------------------------------------------------------------------------------------------------
"""
# todo: EXPERIMENT with different hidden dimension
list_units = np.linspace(20, 100, 5)
list_score = []
list_loss = []
for units in list_units:
    mlp = MyMLP.MyMLP((int(units),), C=10, learning_rate=0.1, max_iter=100, momentum=0.9)
    mlp.fit(X_train, y_train)
    list_score.append(mlp.predict_score(X_test, y_test))
    list_loss.append(mlp.loss())
Visualization.visualize_xyy(list_units, list_loss, list_score, xlabel='Number of hidden units', title="2-layer perceptron")
"""

# ----------------------------------------------------------------------------------------------------------------------
# todo: EXPERIMENT with different epochs
list_epochs = [50, 100, 200, 500, 1000]
list_loss = []
list_score= []
for epoch in list_epochs:
    mlp = MyMLP.MyMLP((100,), C=10, learning_rate=0.1, max_iter=epoch, momentum=0.9)
    mlp.fit(X_train, y_train)
    list_score.append(mlp.predict_score(X_test, y_test))
    list_loss.append(mlp.loss())
Visualization.visualize_xyy(list_epochs, list_loss, list_score, xlabel='Number of epochs', title="2-layer perceptron")
# 500 epochs --> 85%
# 1000 epochs --> 88%
# train thu xem dc max bn %
# ----------------------------------------------------------------------------------------------------------------------
"""
# todo: EXPERIMENT with different input scaling
list_epochs = []
list_loss = []
list_score= []
for epoch in list_epochs:
    mlp = MyMLP.MyMLP((100,), C=10, learning_rate=0.1, max_iter=epoch, momentum=0.9)
    mlp.fit(X_train, y_train)
    list_score.append(mlp.predict_score(X_test, y_test))
    list_loss.append(mlp.loss())
# ----------------------------------------------------------------------------------------------------------------------
# todo: EXPERIMENT with different learning rate

# todo: EXPERIMENT with different momentum

# todo: EXPERIMENT with different activation function

# todo: EXPERIMENT with different number of layers
"""





