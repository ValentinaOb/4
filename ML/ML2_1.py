from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

df = pd.read_csv("ML/pima.csv")  
df = df.apply(lambda col: col.fillna(round(col.mean(),3)))
print(df.head(10))

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy ', accuracy)


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
    print(f"P({cls} | X) = {prob:.3f}")