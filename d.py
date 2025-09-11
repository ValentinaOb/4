import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
from matplotlib.path import Path

from AI_1 import graphic_points 
# 1. Вершини 2D опуклого багатокутника, що лежить у z = 0

#polygon_3d = [(random.uniform(0, 999), random.uniform(0, 999),0), (random.uniform(0, 999), random.uniform(0, 999),0), (random.uniform(0, 999), random.uniform(0, 999),0), (random.uniform(0, 999), random.uniform(0, 999),0), (random.uniform(0, 999), random.uniform(0, 999),0)]
polygon_3d=[(50, 200,0), (440, 50,0), (863, 41,0), (997, 848,0), (41, 952,0)]
polygon_2d=Path([(50, 200), (440, 80), (863, 41), (997, 848), (41, 552)])


# 3. Функція перевірки, чи (x, y) належить 2D опуклому багатокутнику

def is_inside(point, polygon):
    return polygon.contains_point(point)

variable=[]
# 4. Перевірка тільки по (x, y)
for i in range(len(graphic_points)):
    xy_point = graphic_points[i][:2]
    inside = is_inside(xy_point,polygon_2d)
    if inside==True:
        variable.append(graphic_points[i])
    print('\nI ',inside)


# 5. Побудова 3D-графіку
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

poly = Poly3DCollection([polygon_3d], alpha=0.5, facecolor='lightblue', edgecolor='blue')
ax.add_collection3d(poly)

for i in range(len(graphic_points)):
    ax.scatter(graphic_points[i][0], graphic_points[i][1], graphic_points[i][2], color='grey')
    ax.plot([graphic_points[i][0], graphic_points[i][0]], [graphic_points[i][1], graphic_points[i][1]], [0, graphic_points[i][2]],color='lightblue')
for i in range(len(variable)):
    ax.scatter(variable[i][0], variable[i][1], variable[i][2], color='green')
    ax.plot([variable[i][0], variable[i][0]], [variable[i][1], variable[i][1]], [0, variable[i][2]],color='lightblue')    

# Налаштування графіку
ax.set_title("Точка " + ("всередині" if inside else "зовні") + " області (по x,y)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

# Область огляду
ax.set_xlim(0, 999)
ax.set_ylim(0, 999)
ax.set_zlim(0, 999)
plt.tight_layout()
plt.show()

