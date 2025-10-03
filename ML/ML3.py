from statistics import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import itertools
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

def task1():
    df = pd.read_csv("ML/reglab1.csv")
    y = df["z"]
    X = pd.DataFrame({"f1": df["x"] + df["y"]})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(X.head(10))
    lda = LinearRegression()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    
    print("MSE :", mean_squared_error(y_test, y_pred))
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("R^2 :", r2_score(y_test, y_pred))

    X2 = pd.DataFrame({"f2": df["x"]*df["y"] + 2*df["y"] + df["x"]**df["y"]})

    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.3, random_state=42)

    lda2 = LinearRegression()
    lda2.fit(X2_train, y2_train)
    y2_pred = lda2.predict(X2_test)

    print("\n\nMSE :", mean_squared_error(y2_test, y2_pred))
    print("MAE :", mean_absolute_error(y2_test, y2_pred))
    print("R^2 :", r2_score(y2_test, y2_pred))

    X3 = pd.DataFrame({"f2": df["x"]**3 + 1/df["y"]})
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, test_size=0.1, random_state=42)
    lda3 = LinearRegression()
    lda3.fit(X3_train, y3_train)
    y3_pred = lda3.predict(X3_test)

    print("\n\nMSE :", mean_squared_error(y3_test, y3_pred))
    print("MAE :", mean_absolute_error(y3_test, y3_pred))
    print("R^2 :", r2_score(y3_test, y3_pred))

    '''MSE, середня квадратична різниця між фактичними та прогнозованими значеннями,нижчий показник вказує на кращу модель
    A perfect prediction would have an MAE of 0.0. 
    R², коефіцієнт детермінації, представляє частку дисперсії в залежної змінної, яка пояснюється моделлю (1 означає ідеальну модель, 0 - нічого не пояснює'''

def task2():
    df = pd.read_csv("ML/reglab2.csv")
    y = df["y"]
    X = df.drop('y', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    result = best_subset_selection(X_train, y_train)
    subset_model = result.loc[result["rss"].idxmin(), "model"]
    print(subset_model.summary())

    '''t-статистика: показує, наскільки сильно коефіцієнт відрізняється від нуля (в стандартних похибках)
    p-value: якщо нульова гіпотеза H0: Bj=0 - є істинною  
    Малий p-value (< 0.05) → ознака статистично значуща
    Великий p-value → ознака, ймовірно, не впливає, вилучаємо'''

def task3():
    df = pd.read_csv("ML/cygage.csv")
    y = df["Weight"]
    X = df.drop('Weight', axis=1)

    model = LinearRegression()
    y_pred = cross_val_predict(model, X, y, cv=3)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print('MSE ', mse)
    print('R^2 ',r2)

    plt.figure(figsize=(8, 6))
    plt.scatter(X.iloc[:, 0], y, color='blue', label='Actual data')
    plt.plot(X.iloc[:, 0], y_pred, color='red', alpha=0.5, label='Predicted')
    plt.xlabel('X')
    plt.ylabel('Weight')
    plt.title('Cross Validation')
    plt.legend()
    plt.show()

def best_subset_selection(X, y):
    n, d = X.shape
    results = []

    for k in range(1, d+1):
        best_rss = float("inf")
        best_model = None
        best_features = None

        # Усі підмножини ознак розміру k
        for subset in itertools.combinations(range(d), k):
            X_subset = X.iloc[:, list(subset)]
            X_subset = sm.add_constant(X_subset)  # + константу
            model = sm.OLS(y, X_subset).fit()
            rss = ((model.predict(X_subset) - y) ** 2).sum()

            if rss < best_rss:
                best_rss = rss
                best_model = model
                best_features = subset

        results.append({
            "k": k,
            "features": X.columns[list(best_features)],
            "rss": best_rss,
            "model": best_model
        })
    
    print("results:", pd.DataFrame(results))
    return pd.DataFrame(results)

def task4():
    df = pd.read_csv("ML/aligators.csv")
    y = np.array(df["Weight"]).reshape(-1,1)
    x = np.array(df['Length']).reshape(-1,1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)

    poly = PolynomialFeatures(degree=3)
    x_train_poly = poly.fit_transform(x_train)

    poly_reg = LinearRegression()
    poly_reg.fit(x_train_poly, y_train)

    plt.figure(figsize=(10, 6))

    plt.scatter(x_train, y_train, label='Train data')
    plt.scatter(x_test, y_test, label='Test data')

    x_plot = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    plt.plot(x_plot, lin_reg.predict(x_plot), label='Linear')

    x_plot_poly = poly.transform(x_plot)
    plt.plot(x_plot, poly_reg.predict(x_plot_poly), label='Polynomial')

    plt.xlabel("Length")
    plt.ylabel("Weight")
    plt.legend()
    plt.title("Dependence of weight on length")
    plt.show()

def task5():

    data = sm.datasets.longley.load_pandas().data
    X = data.drop(['TOTEMP','POP','YEAR','GNP','UNEMP','ARMED'],axis=1)
    y = data["GNP"]
    print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    lambdas = [10**(-3 + 0.2*i) for i in range(26)]
    train_errors = []
    test_errors = []

    for lam in lambdas:
        model = Ridge(alpha=lam)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_errors.append(mean_squared_error(y_train, y_train_pred))
        test_errors.append(mean_squared_error(y_test, y_test_pred))

    plt.figure(figsize=(8,5))
    plt.plot(lambdas, train_errors, label="Train error", marker='o')
    plt.plot(lambdas, test_errors, label="Test error", marker='s')
    plt.xscale("log")
    plt.xlabel("λ (alpha)")
    plt.ylabel("MSE")
    plt.title("Ridge Regression")
    plt.legend()
    plt.grid(True)
    plt.show()
