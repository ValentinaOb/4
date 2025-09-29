import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("ML/reglab1.csv")
#y = df["z"]
y = (df["z"] > 0.5).astype(int)

X1 = pd.DataFrame({"f1": df["x"] + df["y"]})

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.3, random_state=42)

print(X1.head(10))
lda1 = LinearDiscriminantAnalysis()
lda1.fit(X1_train, y1_train)
y1_pred = lda1.predict(X1_test)

print("Accuracy (x+y):", accuracy_score(y1_test, y1_pred))


X2 = pd.DataFrame({"f2": df["x"]*df["x"] + 2*df["y"] + df["x"]*df["y"]})

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.3, random_state=42)

lda2 = LinearDiscriminantAnalysis()
lda2.fit(X2_train, y2_train)
y2_pred = lda2.predict(X2_test)

print("Accuracy (x*x + 2*y + x*y):", accuracy_score(y2_test, y2_pred))
