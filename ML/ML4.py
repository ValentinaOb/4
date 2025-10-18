
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def task1():
    
    df = pd.read_csv("ML/UCI_Credit_Card.csv")

    X = df.drop('default.payment.next.month', axis=1)
    y = df["default.payment.next.month"]
    
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    threshold = 10

    scores = pd.DataFrame({
        'feature': X.columns,
        'score': selector.scores_
    }).sort_values(by='score')

    print('scores ', scores)
    print('to_drop',to_drop)

    to_drop = scores[scores['score'] < threshold]['feature']
    
    for i in to_drop:
        X = X.drop(i, axis=1)

    #df = pd.get_dummies(df, columns=['Sex']) # One-hot
    print('X ', X.head())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lg=LogisticRegression()
    lg.fit(X_train_scaled, y_train)
    y_pred = lg.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print('\n\nAccuracy ', accuracy)
    print("MSE :", mean_squared_error(y_test, y_pred))
    
def task2():
    df = pd.read_csv("ML/bank-additional-full.csv", sep=";")
    df['y'] = (df['y'] == 'yes').astype(int)
    df['job'] = (df['job'] != 'unemployed').astype(int)
    df['marital'] = (df['marital'] == 'married').astype(int)

    dummies = pd.get_dummies(df['education']).astype(int)
    df = pd.concat([df.drop('education', axis=1), dummies], axis=1)
    
    df['default'] = (df['default'] == 'yes').astype(int)
    df['housing'] = (df['housing'] == 'yes').astype(int)
    
    #print('df ', df)
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
    
    print('\nAccuracy ', accuracy)
    print("MSE :", mean_squared_error(y_test, y_pred))

    #print('X ', X.head())


task2()