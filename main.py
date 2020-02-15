import numpy as np
import MyMLP
import pandas as pd
from sklearn.model_selection import train_test_split

# todo: handle input
df = pd.read_csv('data/train.csv')
labels = np.asarray(df['label'])
#df = df.drop(columns="label")
images = df.drop(columns="label").values

# todo: feature engineering
# rescaling
images = images / 255.0
# split data (8/2)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# todo: implement
mlp = MyMLP.MyMLP((100,), C=10, learning_rate=0.2, max_iter=100, momentum=0.1)
mlp.print()
mlp.fit(X_train, y_train)

# todo: test
pred = mlp.predict(X_test)
score = 100 * np.mean(pred == y_test)
print("Accuracy score: %.2f %%" % score)

