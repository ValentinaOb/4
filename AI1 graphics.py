import matplotlib.pyplot as plt
import numpy as np




N=25 #start_populationt
number_of_mutattion =0
number_of_crossovers_couple=0


x=0
y=0
a=5

x_prev=0
y_prev=0

f=0
z=0

f_array=[]
x_array=[]
y_array=[]


#start of population

for i in range(25):
    
    f+=a/(((x-x_prev)**2)*((y-y_prev)**2)+1)
    x_prev=x
    y_prev=y

    x=np.random.rand()
    y=np.random.rand()

    f_array.append(f)
    x_array.append(x)
    y_array.append(y)

    print('\nf ',f, 'xy', x,' ',y)




fig = plt.figure()
ax = plt.axes(projection='3d')


for i in range(len(x_array)):
    ax.scatter(x_array[i], y_array[i], f_array[i], color='blue')
    ax.plot([x_array[i], x_array[i]], [y_array[i], y_array[i]], [0, f_array[i]],color='lightblue')


ax.set_title('3D Scatter Plot')
plt.show()





'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

color_map = {
    "parent": "blue",
    "child": "green",
    "mutated": "red"
}

for (x, y, z), label in zip(coords, labels):
    ax.scatter(x, y, z, c=color_map[label], label=label)

handles = [plt.Line2D([0], [0], marker='o', color='w', label=key, 
            markerfacecolor=color_map[key], markersize=10) for key in color_map]
ax.legend(handles=handles, loc='upper left')

ax.set_xlabel("Genotype (int)")
ax.set_ylabel("Fitness")
ax.set_zlabel("Generation")
ax.set_title("Visualizing Genetic Algorithm (1 Generation)")

plt.tight_layout()
plt.show()
'''