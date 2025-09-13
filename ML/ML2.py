from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

iris = load_iris(as_frame=True)

df = iris.frame
df['species'] = df['target'].map(dict(zip(range(3), iris.target_names)))

print(df.head())

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


X = iris.data                      # ✅ це DataFrame з правильними назвами
y = iris.target    
feature_names = X.columns 

model = GaussianNB()
model.fit(X, y)                   # Навчання

# 🔍 Приклад нового зразка
sample = {
    'sepal length (cm)': 5.5,
    'sepal width (cm)': 3.0,
    'petal length (cm)': 1.3,
    'petal width (cm)': 0.2
}


X_new = pd.DataFrame([sample], columns=feature_names)

probs = model.predict_proba(X_new)

print('\nProbabilities')
for cls, prob in zip(model.classes_, probs[0]):
    print(f"P({iris.target_names[cls]} | X) = {prob:.4f}")