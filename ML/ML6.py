import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder

def one():
    df = pd.read_csv("ML/monica.csv")
    print(df.head())

    data = df.dropna().copy()
    
    X = data.drop(columns=['outcome'])
    y = data['outcome']

    # categorical columns to numeric via LabelEncoder/get_dummies
    X_proc = pd.get_dummies(X, drop_first=True)
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    print("\nTarget classes:", list(le.classes_))

    X_train, X_test, y_train, y_test = train_test_split(X_proc, y_enc, test_size=0.3, random_state=42, stratify=y_enc)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy {acc:.3f}")
    print("Confusion matrix ", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    plt.figure(figsize=(14,8))
    plot_tree(clf, feature_names=X_proc.columns, class_names=le.classes_, filled=True, max_depth=3)
    plt.title("Decision tree (monica) - truncated to depth")
    plt.show()
  
def two():
    df = pd.read_csv("ML/spam7.csv")
    X = df.drop(columns=["yesno"])
    y = df["yesno"]

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)

    path = clf.cost_complexity_pruning_path(X, y)
    ccp_alphas = path.ccp_alphas
    trees = []

    for ccp in ccp_alphas:
        t = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp)
        t.fit(X, y)
        trees.append(t)

    for i, (alpha, tr) in enumerate(zip(ccp_alphas, trees)):
        print(f"Tree {i}: alpha={alpha:.5f}, leaves={tr.get_n_leaves()}, depth={tr.get_depth()}")

    errors = []
    for tr in trees:
        pred = tr.predict(X)
        errors.append(np.mean(pred != y))

    best_i = np.argmin(errors)
    best_tree = trees[best_i]

    print("Optimal tree ", best_i)
    print("alpha ", ccp_alphas[best_i], "error ", errors[best_i])

    plt.figure(figsize=(12,8))
    plot_tree(best_tree, filled=True)
    plt.show()

def three():
    df = pd.read_csv("ML/nsw74psid1.csv")
    X = df.drop(columns=["re78"])
    y = df["re78"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    rg = DecisionTreeRegressor(random_state=0)
    rg.fit(X_train, y_train)
    pred_rg = rg.predict(X_test)

    ln = LinearRegression()
    ln.fit(X_train, y_train)
    pred_ln = ln.predict(X_test)

    svm = SVR()
    svm.fit(X_train, y_train)
    pred_svm = svm.predict(X_test)

    r2_rg=r2_score(y_test, pred_rg)
    r2_lr=r2_score(y_test, pred_ln)
    r2_svw=r2_score(y_test, pred_svm)
    print("MSE Rg ", mean_squared_error(y_test, pred_rg), "R2 ",r2_rg)
    print("MSE LR ", mean_squared_error(y_test, pred_ln),  "R2 ",r2_lr)
    print("MSE SVM ", mean_squared_error(y_test, pred_svm),  "R2 ",r2_svw)

    if r2_rg<r2_lr and r2_rg<r2_svw:
        print('Opt tree - RG')
    elif r2_lr<r2_rg and r2_lr<r2_svw:
        print('Opt tree - LR')
    else:
        print('Opt tree - SVW')
    # Нижчий RMSE і вищий R^2 — кращий прогноз
    # Дерево вловл нелінійні залежності і взаємодії, але схильне до overfitting. Вик обр / cost-complexity pruning
    # Лінійна модель дає прозору інтерпретацію коефіцієнтів
    # SVM з RBF може уловити складні нелінійності, але потребує налаштування гіперпараметрів

#one()
two()
#three()