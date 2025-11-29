from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils import resample
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import math
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("ML/pima.csv")  
df = df.apply(lambda col: col.fillna(round(col.mean(),3)))
print('\nL ',len(df),'\n',df.head())

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

forest = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=5, min_samples_leaf=3, max_features="sqrt", bootstrap=True, n_jobs=-1)
forest.fit(X_train, y_train)
y_pred=forest.predict(X_test)

print("Forest Accuracy ", accuracy_score(y_test, y_pred))


depths = range(1, 16)
train_accuracy = []
for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    train_accuracy.append(scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(depths, train_accuracy, marker='o', label="Cross-Validation Accuracy")
plt.xlabel('Tree Depth')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Tree Depth vs. Cross-Validated Accuracy')
plt.legend()
plt.grid(True)
plt.show()

depth=train_accuracy.index(max(train_accuracy))+1
print('\nOptimal depth ', depth)

tree = DecisionTreeClassifier(max_depth=depth)
tree.fit(X_train, y_train)
y_pred=tree.predict(X_test)
print("Tree Accuracy ", accuracy_score(y_test, y_pred))
