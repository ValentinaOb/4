from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target


df = iris.frame
df['species'] = df['target'].map(dict(zip(range(3), iris.target_names)))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")



frequency_table = df['species'].value_counts()
percent_table = df['species'].value_counts(normalize=True) * 100

print("\nFrequency Table\n", frequency_table)
print("\n%\n", percent_table.round(2))


def gaussian_likelihood(x, mean, std):
    exponent = np.exp(- ((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent


print('\n')
feature = 'sepal length (cm)'
value = 5.5

for species in df['species'].unique():
    group = df[df['species'] == species]
    mean = group[feature].mean()
    std = group[feature].std()
    prob = gaussian_likelihood(value, mean, std)
    print(f"P({feature}={value} | {species}) = {prob:.4f}")


#
X = iris.data 
y = iris.target    
feature_names = X.columns

model = GaussianNB()
model.fit(X, y)

# New element
sample = {
    'sepal length (cm)': 5.5,
    'sepal width (cm)': 3.9,
    'petal length (cm)': 2.7,
    'petal width (cm)': 1.2
}

X_new = pd.DataFrame([sample], columns=feature_names)

probs = model.predict_proba(X_new)

print('\nProbabilities')
for cls, prob in zip(model.classes_, probs[0]):
    print(f"P({iris.target_names[cls]} | X) = {prob:.3f}")
