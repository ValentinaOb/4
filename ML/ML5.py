import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("ML/svmdata1.csv")
df_test = pd.read_csv("ML/svmdata1.csv")
#df = pd.get_dummies(df, columns=['Color']) # One-hot       Not use for one()
#df_test = pd.get_dummies(df_test, columns=['Color']) # One-hot

X_train = df.iloc[:, :-1].values
y_train = df.iloc[:, -1].values
X_test  = df_test.iloc[:, :-1].values
y_test  = df_test.iloc[:, -1].values

scaler = StandardScaler().fit(np.vstack([X_train, X_test]))
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

def plot_decision_boundary(clf, X, y, ax=None):
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_full = grid
    
    Z = clf.predict(grid_full)
    try:
        Z = Z.astype(float)
    except:
        from sklearn.preprocessing import LabelEncoder
        Z = LabelEncoder().fit_transform(Z)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:,0], X[:,1], c=y, edgecolor='k', s=40)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    return ax

def one():
    svc1 = SVC(C=1.0, kernel='linear')
    svc1.fit(X_train_s, y_train)

    n_sv = svc1.support_vectors_.shape[0]
    print('Numb of support vectors ',n_sv)

    y_test_pred  = svc1.predict(X_test_s)
    test_acc  = accuracy_score(y_test, y_test_pred)
    print('Accuracy ', test_acc)

    plot_decision_boundary(svc1, X_test_s[:, :2], y_test)
    sv_idx = svc1.support_
    plt.scatter(X_test_s[sv_idx,0], X_test_s[sv_idx,1], s=120, facecolors='none', edgecolors='r', label='support vectors')
    plt.legend()
    plt.show()

def two():
    C_values = np.logspace(0, 8, 10)
    results = []
    for C in C_values:
        clf = SVC(C=C, kernel='linear')
        clf.fit(X_train_s, y_train)
        tr_err = 1 - accuracy_score(y_train, clf.predict(X_train_s))
        te_err = 1 - accuracy_score(y_test, clf.predict(X_test_s))
        nsv = clf.support_vectors_.shape[0]
        results.append((C, tr_err, te_err, nsv))

    res_df = pd.DataFrame(results, columns=["C","train_err","test_err","n_support"])
    zero_train = res_df[res_df.train_err == 0]
    C_min_zero_train = zero_train.iloc[0].C
    print('Min C without error Train, ', C_min_zero_train)

    best_idx = res_df.test_err.idxmin()
    best_row = res_df.loc[best_idx]
    print('Optimal in terms of minimum errors ',best_row.C,'test error ',best_row.test_err, 'train error ',best_row.train_err, 'n_sv ',int(best_row.n_support))

    zero_test = res_df[res_df.test_err == 0]
    if not zero_test.empty:
        C_min_zero_test = zero_test.iloc[0].C
        print('Min C without error ',C_min_zero_test)
    else:
        print("Not found")

    plt.figure(figsize=(8,5))
    plt.semilogx(res_df.C, res_df.train_err, label="train_err")
    plt.semilogx(res_df.C, res_df.test_err, label="test_err")
    plt.xlabel("C")
    plt.ylabel("Error rate")
    plt.title("Train/Test error and C")
    plt.legend()
    plt.show()

def three():
    kernels = ["poly", "rbf", "sigmoid"]
    poly_degrees = [2,3,4,5]
    best_overall = {"kernel": None, "params": None, "test_err": 1.0}

    records = []
    for kernel in kernels:
        if kernel == "poly":
            for d in poly_degrees:
                clf = SVC(C=1.0, kernel='poly', degree=d)
                clf.fit(X_train_s, y_train)
                te = 1 - accuracy_score(y_test, clf.predict(X_test_s))
                tr = 1 - accuracy_score(y_train, clf.predict(X_train_s))
                nsv = clf.support_vectors_.shape[0]
                records.append(("poly", d, tr, te, nsv))
                if te < best_overall["test_err"]:
                    best_overall.update(kernel="poly", params={"degree":d, "C":1.0}, test_err=te)
        else:
            clf = SVC(C=1.0, kernel=kernel)
            clf.fit(X_train_s, y_train)
            te = 1 - accuracy_score(y_test, clf.predict(X_test_s))
            tr = 1 - accuracy_score(y_train, clf.predict(X_train_s))
            nsv = clf.support_vectors_.shape[0]
            records.append((kernel, np.nan, tr, te, nsv))
            if te < best_overall["test_err"]:
                best_overall.update(kernel=kernel, params={"C":1.0}, test_err=te)

    rec_df = pd.DataFrame(records, columns=["kernel","degree","train_err","test_err","n_sv"])
    print("Res ",rec_df)
    print("\nbest_overall ",best_overall)

    best_kernel = best_overall["kernel"]
    
    param_grid = {"C": np.logspace(-3, 3, 13)}
    if best_kernel == "poly":
        param_grid["degree"] = poly_degrees
        svc = SVC(kernel="poly")
    else:
        svc = SVC(kernel=best_kernel)
    gs = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train_s, y_train)
    print("GridSearch params ", gs.best_params_, "- cv score ", gs.best_score_)
    final = gs.best_estimator_
    accuracy=accuracy_score(y_test, final.predict(X_test_s))
    print("Accuracy ", accuracy)

def four():
    eps_values = np.linspace(0.0, 1.0, 21)
    mse_train = []
    for eps in eps_values:
        svr = SVR(C=1.0, kernel='rbf', epsilon=eps)
        svr.fit(X_train_s, y_train)
        pred_tr = svr.predict(X_train_s)
        mse = mean_squared_error(y_train, pred_tr)
        mse_train.append(mse)
        
    df_eps = pd.DataFrame({"epsilon": eps_values, "mse_train": mse_train})
    print(df_eps)
    
    plt.figure(figsize=(8,5))
    plt.plot(df_eps["epsilon"], df_eps["mse_train"], marker='o')
    plt.xlabel("Epsilon")
    plt.ylabel("MSE train")
    plt.title("SVR. MSE and ε")
    plt.grid(True)
    plt.show()

two()