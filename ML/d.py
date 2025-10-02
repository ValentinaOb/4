import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# --- прикладні дані (зріст у см, вага у кг) ---
height = np.array([150, 155, 160, 165, 170, 175, 180, 185]).reshape(-1, 1)
weight = np.array([45, 50, 54, 59, 65, 72, 80, 90])

# --- 1. Побудуємо лінійну регресію ---
lin_reg = LinearRegression()
lin_reg.fit(height, weight)
weight_pred_lin = lin_reg.predict(height)

# --- 2. Поліноміальна регресія (квадратична) ---
poly = PolynomialFeatures(degree=2)
height_poly = poly.fit_transform(height)
poly_reg = LinearRegression()
poly_reg.fit(height_poly, weight)
weight_pred_poly = poly_reg.predict(height_poly)

# --- Графік ---
plt.scatter(height, weight, color="blue", label="Фактичні дані")
plt.plot(height, weight_pred_lin, color="red", label="Лінійна модель")
plt.plot(height, weight_pred_poly, color="green", label="Поліноміальна модель (deg=2)")

plt.xlabel("Зріст (см)")
plt.ylabel("Вага (кг)")
plt.legend()
plt.title("Залежність ваги від зросту")
plt.show()
