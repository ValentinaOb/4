from statistics import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import itertools
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

def task1():
    df = pd.read_csv("ML/reglab1.csv")
    #y = df["z"]
    y = (df["z"] > 0.5).astype(int)
    print('y',y)
    X1 = pd.DataFrame({"f1": df["x"] + df["y"]})

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.3, random_state=42)

    print(X1.head(10))
    lda1 = LinearDiscriminantAnalysis()
    lda1.fit(X1_train, y1_train)
    y1_pred = lda1.predict(X1_test)

    print("Accuracy (x+y):", accuracy_score(y1_test, y1_pred))

    X2 = pd.DataFrame({"f2": df["x"]*df["y"] + 2*df["y"] + df["x"]**df["y"]})

    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.3, random_state=42)

    lda2 = LinearDiscriminantAnalysis()
    lda2.fit(X2_train, y2_train)
    y2_pred = lda2.predict(X2_test)

    print("Accuracy (x*y + 2*y + x^y):", accuracy_score(y2_test, y2_pred))

def task2():
    df = pd.read_csv("ML/reglab2.csv")
    y = (df["y"] > 0.5).astype(int)
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
        
    print("MSE:", mse)
    print("R^2:", r2)
    '''MSE, середня квадратична різниця між фактичними та прогнозованими значеннями,нижчий показник вказує на кращу модель
    R², коефіцієнт детермінації, представляє частку дисперсії в залежної змінної, яка пояснюється моделлю (1 означає ідеальну модель, 0 - нічого не пояснює'''

    plt.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Actual data')
    plt.plot(X_test.iloc[:, 0], y_pred, color='red', linewidth=2, label='Regression line')
    plt.xlabel('Independent Variable (X)')
    plt.ylabel('Dependent Variable (y)')
    plt.title('Simple Linear Regression')
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

    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    weight_pred_lin = lin_reg.predict(x)

    poly = PolynomialFeatures(degree=10)
    height_poly = poly.fit_transform(x)
    poly_reg = LinearRegression()
    poly_reg.fit(height_poly, y)
    weight_pred_poly = poly_reg.predict(height_poly)

    plt.scatter(x, y, label="Data")
    plt.plot(x, weight_pred_lin, label="Linear")
    plt.plot(x, weight_pred_poly, label="Polynomial")

    plt.xlabel("Length")
    plt.ylabel("Weight")
    plt.legend()
    plt.title("Dependence of weight on length")
    plt.show()

def task6():
    longley = sm.datasets.get_rdataset('longley')
    X = longley.data.drop('Population', 'Year', axis=1)
    y = longley.data['Year']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. Scale the features (important for regularization methods)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # A higher alpha means stronger regularization (coefficients shrink more)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)

    y_pred = ridge_model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2) Score: {r2:.2f}")
    print(f"Model Coefficients: {ridge_model.coef_}")
    print(f"Model Intercept: {ridge_model.intercept_}")

def task5():

    data = sm.datasets.longley.load_pandas().data
    print(data)
    X = data.drop(['TOTEMP','POP'],axis=1)
    y = data["TOTEMP"]

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


task5()