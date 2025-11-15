import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train,y_train,X_test,y_test,X_train_s,X_test_s=None,None,None,None,None,None

def data(n):
    global X_train,y_train,X_test,y_test,X_train_s,X_test_s
    if n == 1:    
        df = pd.read_csv("ML/svmdata1.csv")
        df_test = pd.read_csv("ML/svmdata1.csv")
        #df = pd.get_dummies(df, columns=['Color']) # One-hot       Not use for one()
        #df_test = pd.get_dummies(df_test, columns=['Color']) # One-hot
    elif n == 2:    
        df = pd.read_csv("ML/svmdata2.csv")
        df_test = pd.read_csv("ML/svmdata2.csv")
    elif n == 3:    
        df = pd.read_csv("ML/svmdata3.csv")
        df_test = pd.read_csv("ML/svmdata3.csv")
    else:
        df = pd.read_csv("ML/svmdata4.csv")
        df_test = pd.read_csv("ML/svmdata4.csv")
        df = pd.get_dummies(df, columns=['Colors']) # One-hot       Not use for one()
        df_test = pd.get_dummies(df_test, columns=['Colors']) # One-hot

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
    global X_train,y_train,X_test,y_test,X_train_s,X_test_s
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
    global X_train,y_train,X_test,y_test,X_train_s,X_test_s
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
    global X_train,y_train,X_test,y_test,X_train_s,X_test_s
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
    global X_train,y_train,X_test,y_test,X_train_s,X_test_s
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

def five():
    df = pd.read_csv("ML/bank-additional-full.csv", sep=";")
    df['y'] = (df['y'] == 'yes').astype(int)
    df['job'] = (df['job'] != 'unemployed').astype(int)
    df['marital'] = (df['marital'] == 'married').astype(int)

    dummies = pd.get_dummies(df['education']).astype(int)
    df = pd.concat([df.drop('education', axis=1), dummies], axis=1)
    
    df['default'] = (df['default'] == 'yes').astype(int)
    df['housing'] = (df['housing'] == 'yes').astype(int)
    
    X = df.drop(['y', 'loan', 'day', 'month','poutcome', 'contact'], axis=1)
    y = df["y"]

    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    threshold = 10

    scores = pd.DataFrame({
        'feature': X.columns,
        'score': selector.scores_
    }).sort_values(by='score')    

    to_drop = scores[scores['score'] < threshold]['feature']
    
    print('\nscores ', scores)
    print('to_drop',to_drop)
    
    for i in to_drop:
        X = X.drop(i, axis=1)

    print('\n', X.head(5))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lg=LogisticRegression()
    lg.fit(X_train_scaled, y_train)
    y_pred = lg.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print('\nLogisticRegression Accuracy ', accuracy)
    print("MSE :", mean_squared_error(y_test, y_pred))

    svc1 = SVC(C=1.0, kernel='linear')
    svc1.fit(X_train_scaled, y_train)

    n_sv = svc1.support_vectors_.shape[0]
    print('\nNumb of support vectors ',n_sv)

    y_test_pred  = svc1.predict(X_test_scaled)
    test_acc  = accuracy_score(y_test, y_test_pred)
    print('SVC Accuracy ', test_acc)

    # --- 2. Виявлення аномалій за допомогою Z-score ---


def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return (z_scores > threshold)

def detect_outliers_iqr(data, factor=1.5):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return (data < lower_bound) | (data > upper_bound)

def six():
    df = pd.read_csv("ML/pima.csv")  
    print("Data")
    print(df.head())

    df = df.apply(lambda col: col.fillna(round(col.mean(),3)))

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    svc1 = SVC(C=1.0, kernel='linear')
    svc1.fit(X_train_s, y_train)
    n_sv = svc1.support_vectors_.shape[0]
    print('\nNumb of support vectors ',n_sv)
    y_test_pred  = svc1.predict(X_test_s)
    test_acc  = accuracy_score(y_test, y_test_pred)
    print('IQR Accuracy ', test_acc)

    # Outliers
    cols = df.select_dtypes(include=[np.number]).columns

    df1=df

    for col in cols:
        outliers = detect_outliers_iqr(df[col])
        count_outliers = outliers.sum()
        if count_outliers!=0:
            print(f"\nIQR Analyse {col}")
            print(f"Detect {count_outliers} outliers")

        median_value = df[col].median()
        df.loc[outliers, col] = median_value
        
        #
        outliers = detect_outliers_zscore(df[col])
        count_outliers = outliers.sum()
        if count_outliers!=0:
            print(f"\nZSCORE Analyse {col}")
            print(f"Detect {count_outliers} outliers")

        median_value = df1[col].median()
        df1.loc[outliers, col] = median_value

    
    print("\nData After IQR")
    print(df.head())

    print("\nData After ZSCORE")
    print(df1.head())
    
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    svc1 = SVC(C=1.0, kernel='linear')
    svc1.fit(X_train_s, y_train)

    n_sv = svc1.support_vectors_.shape[0]
    print('\nNumb of support vectors ',n_sv)

    y_test_pred  = svc1.predict(X_test_s)
    test_acc  = accuracy_score(y_test, y_test_pred)
    print('IQR Accuracy ', test_acc)

    #
    X = df1.drop('Outcome', axis=1)
    y = df1['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    svc1 = SVC(C=1.0, kernel='linear')
    svc1.fit(X_train_s, y_train)

    n_sv = svc1.support_vectors_.shape[0]
    print('\nNumb of support vectors ',n_sv)

    y_test_pred  = svc1.predict(X_test_s)
    test_acc  = accuracy_score(y_test, y_test_pred)
    print('ZSCORE Accuracy ', test_acc)

data(1)
six()

