from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("ML/pima.csv")  

print(df.head())


frequency_table = df['Outcome'].value_counts()
percent_table = df['Outcome'].value_counts(normalize=True) * 100

print('\nFrequency Table\n', frequency_table)
print('\n%\n', percent_table.round(2))


def gaussian_likelihood(x, mean, std):
    exponent = np.exp(- ((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent


print('\n')
feature = 'DiabetesPedigreeFunction'
value = 0.485

for outcome in df['Outcome'].unique():
    group = df[df['Outcome'] == outcome]
    mean = group[feature].mean()
    std = group[feature].std()
    prob = gaussian_likelihood(value, mean, std) # щільність ймовірності норм розподілу
    print(f"P({feature}={value} | Outcome={outcome}) = {prob:.4f}")


#
X = df 
y = df['Outcome']  
feature_names = X.columns

model = GaussianNB()
model.fit(X, y)

# New element
sample = {
    'Pregnancies':2,
    'Glucose': 99,
    'BloodPressure':54,
    'SkinThickness':16,
    'Insulin':17,
    'BMI':24.6,
    'DiabetesPedigreeFunction':0.145,
    'Age':81,
    'Outcome':1
}

X_new = pd.DataFrame([sample], columns=feature_names)

probs = model.predict_proba(X_new)

print('\nProbabilities')
for cls, prob in zip(model.classes_, probs[0]):
    print(f"P({df['Outcome'].unique()[cls]} | X) = {prob:.3f}")