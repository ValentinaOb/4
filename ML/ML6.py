import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score

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

    print(df.head())
    if 'yesno' not in df.columns:
        raise ValueError("Dataset 'spam7' не містить стовпця 'yesno' — перевірте назви колонок.")

    data = df.dropna().copy()
    X = data.drop(columns=['yesno'])
    y = data['yesno']

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    X_proc = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X_proc, y_enc, test_size=0.3, random_state=42, stratify=y_enc)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    print("Unpruned tree Accuracy", accuracy_score(y_test, clf.predict(X_test)))

    # Cost-complexity pruning path
    path = clf.cost_complexity_pruning_path(X_train, y_train)  # returns ccp_alphas and impurities
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    print(f"Found {len(ccp_alphas)} ccp_alpha candidates")

    # Build trees for each alpha, evaluate via cross-val misclassification error (1 - accuracy)
    clfs = []
    alpha_list = []
    cv_scores_mean = []
    cv_scores_std = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for ccp_alpha in ccp_alphas:
        clf_temp = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        # use cross_val_score for accuracy
        scores = cross_val_score(clf_temp, X_train, y_train, cv=kf, scoring='accuracy')
        clfs.append(clf_temp)
        alpha_list.append(ccp_alpha)
        cv_scores_mean.append(scores.mean())
        cv_scores_std.append(scores.std())

    results = pd.DataFrame({
        'ccp_alpha': alpha_list,
        'cv_accuracy_mean': cv_scores_mean,
        'cv_accuracy_std': cv_scores_std
    })
    print("\nSequence of trees (alpha vs CV accuracy):")
    print(results.sort_values('ccp_alpha').reset_index(drop=True))

    # Plot accuracy vs alpha
    plt.figure(figsize=(8,5))
    plt.errorbar(results['ccp_alpha'], results['cv_accuracy_mean'], yerr=results['cv_accuracy_std'], marker='o', linestyle='-')
    plt.xscale('log' if (results['ccp_alpha'] > 0).any() else 'linear')
    plt.xlabel('ccp_alpha (pruning strength)')
    plt.ylabel('CV accuracy (mean)')
    plt.title('Cost-complexity pruning - alpha vs CV accuracy')
    plt.show()

    # Choose optimal alpha (max cv accuracy / min misclassification
    best_idx = int(np.argmax(results['cv_accuracy_mean']))
    best_alpha = results.loc[best_idx, 'ccp_alpha']
    print(f"\nOptimal alpha by CV Accuracy {best_alpha} (index {best_idx}), CV accuracy {results.loc[best_idx,'cv_accuracy_mean']:.4f}")

    final_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
    final_clf.fit(X_train, y_train)
    y_pred = final_clf.predict(X_test)
    print("Pruned tree Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix\n", confusion_matrix(y_test, y_pred))
    print("Classification report\n", classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Оптимальне дерево - мінімізує помилку на валідації / максимізує accuracy

def three():
    df = pd.read_csv("ML/nsw74psid1.csv")
    print(df.head())

    data = df.dropna().copy()
    X = data.drop(columns=['re78'])
    y = data['re78'].astype(float)

    X_proc = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_proc)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    # Regression tree
    regr_tree = DecisionTreeRegressor(random_state=42)
    regr_tree.fit(X_train, y_train)
    y_pred_tree = regr_tree.predict(X_test)
    rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
    r2_tree = r2_score(y_test, y_pred_tree)
    print(f"Tree RMSE: {rmse_tree:.2f}, R2: {r2_tree:.3f}")

    # LR
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred_lin = lin.predict(X_test)
    rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
    r2_lin = r2_score(y_test, y_pred_lin)
    print(f"Linear RMSE: {rmse_lin:.2f}, R2: {r2_lin:.3f}")

    # SVM
    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test)
    rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
    r2_svr = r2_score(y_test, y_pred_svr)
    print(f"SVR RMSE: {rmse_svr:.2f}, R2: {r2_svr:.3f}")

    # Plot predicted vs actual for test set
    plt.figure(figsize=(15,4))
    for i, (y_pred, title) in enumerate(zip([y_pred_tree, y_pred_lin, y_pred_svr], ['Tree', 'Linear', 'SVR'])):
        plt.subplot(1,3,i+1)
        plt.scatter(y_test, y_pred, alpha=0.6)
        mx = max(y_test.max(), np.max(y_pred))
        mn = min(y_test.min(), np.min(y_pred))
        plt.plot([mn, mx], [mn, mx], linestyle='--')
        plt.xlabel('Actual re78')
        plt.ylabel('Predicted re78')
        plt.title(f'{title} (RMSE={np.sqrt(mean_squared_error(y_test,y_pred)):.2f}, R2={r2_score(y_test,y_pred):.3f})')
    plt.tight_layout()
    plt.show()

    # Нижчий RMSE і вищий R^2 — кращий прогноз
    # Дерево вловл нелінійні залежності і взаємодії, але схильне до overfitting. Вик обр / cost-complexity pruning
    # Лінійна модель дає прозору інтерпретацію коефіцієнтів
    # SVM з RBF може уловити складні нелінійності, але потребує налаштування гіперпараметрів

#one()
two()
#three()