import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
from tkinter import *

#1
polygon_3d=[(50, 200), (440, 50), (863, 41), (997, 848), (41, 952),(50, 200)]
polygon_2d=Path([(50, 200), (440, 80), (863, 41), (997, 848), (41, 552)])

#2
P=5 #population #50
C=2 #crosover #20
M=1 #mutation #15
D=3 #random members #10


population = []
cross=[]
mut=[]
rand=[]
graphic_points=[]
graphic_crossover=[]
graphic_mutation=[]
grahp_elements=[]

indiv=[]
f,x_prev,y_prev=0,0,0
a=1
r_f=0
r_m=0

#3

def is_inside(point, polygon):
    return polygon.contains_point(point)


def individual ():
    '''xc=random.randrange(0,999)
    xd=random.randrange(0,999)
    
    yc=random.randrange(0,999)
    yd=random.randrange(0,999)'''
    global f, x_prev, y_prev
    inside=False
    while inside==False:
        x = round(random.uniform(0.0, 999.999), 3)
        y = round(random.uniform(0.0, 999.999), 3)
    
        #   Check D                        !!!
            
        inside = is_inside([x,y],polygon_2d)
        if inside==True:
            inside=True
                
            f+=a/(((x-x_prev)**2)+((y-y_prev)**2)+1)
            x_prev=x
            y_prev=y

            xc = int(x)
            xd = int(str((round(x - int(x),3))).replace("0.",""))
            yc = int(y)
            yd = int(str((round(y - int(y),3))).replace("0.",""))

            return [xc,xd,yc,yd,f]

def crossover (f,m):
    cross.clear()
    r_point=random.randrange(0,4)
    print('\n\nf, m, r_point ',f,m, r_point)

    ind1=f[:r_point] + m[r_point:]
    ind2=m[:r_point] + f[r_point:]
    print('\nind ',ind1,ind2)
    return ind1, ind2

def mutation (ind):
    mut.clear()
    r_point=random.randrange(0,4)
    new_random=random.randint(0,999)
    new_ind=ind[:r_point] +[new_random]+ ind[r_point+1:] #???
    

    print('\n\nf, r_point ',ind, r_point)
    print('\nnew_ind ',new_ind)
    return new_ind

def sorting ():
    population.sort(key=lambda x: x[4], reverse=True)
    print("\nSorted ", population)

def selection ():
    global grahp_elements
    grahp_elements.sort(key=lambda x: x[2], reverse=True)
    
    print("\nPopulation ", population)
    
    for i in range(P):
        xc,xd = int(grahp_elements[i][0]), int(str((round(grahp_elements[i][0] - int(grahp_elements[i][0]),3))).replace("0.",""))
        yc,yd = int(grahp_elements[i][1]), int(str((round(grahp_elements[i][1] - int(grahp_elements[i][1]),3))).replace("0.",""))
        f=grahp_elements[i][2]
        population.append([xc,xd,yc,yd,f])
    print("\nNew population ", population)

def algor():
    global population, r_f, r_m

    population = [individual() for i in range(P)]
    print('\nPopulation ',population)

    #4
    sorting()

    #5
    for i in range(C):
        while r_f==r_m:
            r_f=random.randrange(0,P)
            r_m=random.randrange(0,P)
        ind1, ind2=crossover(population[r_f],population[r_m])
        cross.append(ind1)
        cross.append(ind2)

    print('Crossover ', cross)

    #6
    for i in range(M):
        r_ind=random.randrange(0,P)
        ind=mutation(population[r_ind])
        mut.append(ind)

    #6.5 Add D / Random
    r_d = [individual() for i in range(D)]
    print('\nD ',r_d)

    prepare_for_graphic()

    #7
    selection()

def prepare_for_graphic(): #connect
    global population, cross, mut
    global graphic_points, graphic_crossover, graphic_mutation, grahp_elements
    graphic_points, graphic_crossover, graphic_mutation, grahp_elements=[],[],[],[]
    

    for i in range(len(population)):
        x=float(str(population[i][0])+'.'+str(population[i][1]))
        y=float(str(population[i][2])+'.'+str(population[i][3]))
        f=population[i][4]
        graphic_points.append([x,y,f])
        
    for i in range(len(cross)):
        x=float(str(cross[i][0])+'.'+str(cross[i][1]))
        y=float(str(cross[i][2])+'.'+str(cross[i][3]))
        f=cross[i][4]
        graphic_crossover.append([x,y,f])

    for i in range(len(mut)):
        x=float(str(mut[i][0])+'.'+str(mut[i][1]))
        y=float(str(mut[i][2])+'.'+str(mut[i][3]))
        f=mut[i][4]
        graphic_mutation.append([x,y,f])

    grahp_elements=graphic_points+graphic_crossover+graphic_mutation
    
def grahp():
    global graphic_points, graphic_crossover, graphic_mutation, grahp_elements

    ax = plt.axes(projection='3d')
    print('!!!          ',len(graphic_points))

    # Polygon
    for i in range(len(polygon_3d)):
        plt.scatter(polygon_3d[i][0],polygon_3d[i][1],0)
    #

    for i in range(len(graphic_points)):
        ax.scatter(graphic_points[i][0], graphic_points[i][1], graphic_points[i][2], color='grey')
        ax.plot([graphic_points[i][0], graphic_points[i][0]], [graphic_points[i][1], graphic_points[i][1]], [0, graphic_points[i][2]],color='lightblue')
    
    for i in range(len(graphic_crossover)):
        ax.scatter(graphic_crossover[i][0], graphic_crossover[i][1], graphic_crossover[i][2], color='blue')
        ax.plot([graphic_crossover[i][0], graphic_crossover[i][0]], [graphic_crossover[i][1], graphic_crossover[i][1]], [0, graphic_crossover[i][2]],color='lightblue')

    for i in range(len(graphic_mutation)):
        ax.scatter(graphic_mutation[i][0], graphic_mutation[i][1], graphic_mutation[i][2], color='yellow')
        ax.plot([graphic_mutation[i][0], graphic_mutation[i][0]], [graphic_mutation[i][1], graphic_mutation[i][1]], [0, graphic_mutation[i][2]],color='lightblue')

    x_coords, y_coords = zip(*polygon_3d)
    ax.plot(x_coords, y_coords,color ='pink')
    

    #max
    max=0
    for i in range(len(grahp_elements)):
        if max< grahp_elements[i][2]:
            max=grahp_elements[i][2]
            max_z_index=i
    #ax.scatter(graphic_points[max_z_index][0], graphic_points[max_z_index][1], graphic_points[max_z_index][2], color='red')
    ax.text(graphic_points[max_z_index][0], graphic_points[max_z_index][1], graphic_points[max_z_index][2],  'm', size=10, color='r') 
    '''
    for i in range(len(graphic_points)):
        if max< graphic_points[i][2]:
            max=graphic_points[i][2]
            max_z_index=[0,i]
    for i in range(len(graphic_crossover)):
        if max< graphic_crossover[i][2]:
            max=graphic_crossover[i][2]
            max_z_index=[1,i]
    for i in range(len(graphic_mutation)):
        if max< graphic_mutation[i][2]:
            max=graphic_mutation[i][2]
            max_z_index=[2,i]
    if max_z_index[0,0]==0:
        ax.scatter(graphic_points[max_z_index[0,1]][0], graphic_points[max_z_index[0,1]][1], graphic_points[max_z_index[0,1]][2], color='red')
    elif max_z_index[0,0]==1:
        ax.scatter(graphic_crossover[max_z_index[0,1]][0], graphic_crossover[max_z_index[0,1]][1], graphic_crossover[max_z_index[0,1]][2], color='red')
    '''
    #
    


    ax.set_title('3D Scatter Plot')
    plt.show()


def main():
    global P, C, M, D

    master = Tk()
    Label(master, text='Population').grid(row=0)
    Label(master, text='Crossover').grid(row=1)
    Label(master, text='Mutation').grid(row=2)
    Label(master, text='D').grid(row=3)
    e1 = Entry(master)
    e2 = Entry(master)
    e1.grid(row=0, column=1)
    e2.grid(row=1, column=1)
    mainloop()


#main()
algor()
prepare_for_graphic()
grahp()