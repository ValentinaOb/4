import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
from tkinter import *
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
r_d=[]
graphic_points=[]
graphic_crossover=[]
graphic_mutation=[]
grahp_elements=[]
graphic_r_d=[]

indiv=[]
f,x_prev,y_prev=0,0,0
a=1
r_f=0
r_m=0

ax=None

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

def crossover (fa,m):
    global f
    r_point=random.randrange(0,4)
    print('\n\nf, m, r_point ',f,m, r_point)

    ind1=fa[:r_point] + m[r_point:]
    ind2=m[:r_point] + fa[r_point:]

    x_prev=float(str(fa[0])+'.'+str(fa[1]))
    y_prev=float(str(fa[2])+'.'+str(fa[3]))
    x=float(str(ind1[0])+'.'+str(ind1[1]))
    y=float(str(ind1[2])+'.'+str(ind1[3]))
    
    f+=a/(((x-x_prev)**2)+((y-y_prev)**2)+1)
    ind1[4]=f


    x_prev=float(str(m[0])+'.'+str(m[1]))
    y_prev=float(str(m[2])+'.'+str(m[3]))
    x=float(str(ind2[0])+'.'+str(ind2[1]))
    y=float(str(ind2[2])+'.'+str(ind2[3]))
    
    f+=a/(((x-x_prev)**2)+((y-y_prev)**2)+1)
    ind2[4]=f

    print('\nind ',ind1,ind2)
    return ind1, ind2

def mutation (ind):    
    global f
    r_point=random.randrange(0,4)
    new_random=random.randint(0,999)
    new_ind=ind[:r_point] +[new_random]+ ind[r_point+1:] #???
    
    x_prev=float(str(ind[0])+'.'+str(ind[1]))
    y_prev=float(str(ind[2])+'.'+str(ind[3]))
    x=float(str(new_ind[0])+'.'+str(new_ind[1]))
    y=float(str(new_ind[2])+'.'+str(new_ind[3]))
    
    f+=a/(((x-x_prev)**2)+((y-y_prev)**2)+1)
    new_ind[4]=f

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
    population.clear()
    
    for i in range(P):
        xc,xd = int(grahp_elements[i][0]), int(str((round(grahp_elements[i][0] - int(grahp_elements[i][0]),3))).replace("0.",""))
        yc,yd = int(grahp_elements[i][1]), int(str((round(grahp_elements[i][1] - int(grahp_elements[i][1]),3))).replace("0.",""))
        f=grahp_elements[i][2]
        population.append([xc,xd,yc,yd,f])
    print("\nNew population ", population)

def algor(btn):
    global population, r_f, r_m,r_d,grahp_elements,graphic_crossover,graphic_mutation,graphic_points,graphic_r_d

    if btn==0:
        population = [individual() for i in range(P)]
    else:
        print('\n\n\nHere\n\n')
        mut.clear()
        cross.clear()
        r_d.clear()
        grahp_elements.clear()
        graphic_crossover.clear()
        graphic_mutation.clear()
        graphic_points.clear()
        graphic_r_d.clear()
        
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

    #prepare_for_graphic()

    #
    prepare_for_graphic()
    grahp()


    #7
    selection()


    
    

    #

def prepare_for_graphic(): #connect
    global population, cross, mut, r_d
    global graphic_points, graphic_crossover, graphic_mutation, grahp_elements, graphic_r_d
    graphic_points, graphic_crossover, graphic_mutation, grahp_elements, graphic_r_d =[],[],[],[],[]
    

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

    for i in range(len(r_d)):
        x=float(str(r_d[i][0])+'.'+str(r_d[i][1]))
        y=float(str(r_d[i][2])+'.'+str(r_d[i][3]))
        f=r_d[i][4]
        graphic_r_d.append([x,y,f])

    grahp_elements=graphic_points+graphic_crossover+graphic_mutation+graphic_r_d
    
def grahp():
    global graphic_points, graphic_crossover, graphic_mutation, grahp_elements, ax

    ax = plt.axes(projection='3d')
    
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
        ax.scatter(graphic_mutation[i][0], graphic_mutation[i][1], graphic_mutation[i][2], color='orange')
        ax.plot([graphic_mutation[i][0], graphic_mutation[i][0]], [graphic_mutation[i][1], graphic_mutation[i][1]], [0, graphic_mutation[i][2]],color='lightblue')

    #
    for i in range(len(graphic_r_d)):
        ax.scatter(graphic_r_d[i][0], graphic_r_d[i][1], graphic_r_d[i][2], color='black')
        ax.plot([graphic_r_d[i][0], graphic_r_d[i][0]], [graphic_r_d[i][1], graphic_r_d[i][1]], [0, graphic_r_d[i][2]],color='lightblue')

    #
    x_coords, y_coords = zip(*polygon_3d)
    ax.plot(x_coords, y_coords,color ='pink')
    

    #max
    max=0
    for i in range(len(grahp_elements)):
        if max< grahp_elements[i][2]:
            max=grahp_elements[i][2]
            max_z_index=i
    ax.text(grahp_elements[max_z_index][0], grahp_elements[max_z_index][1], grahp_elements[max_z_index][2],  'm', size=10, color='r') 
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
    
def update_canvas():    
    
    fields_gr['canvas'].draw()
    
    for item in fields_gr['table'].get_children():
        fields_gr['table'].delete(item)
    k=0
    for i in grahp_elements:
        #for x, y, f in zip(i[0], i[1],i[2]):
        if k==0:
            fields_gr['table'].insert("", "end", values=('', 'Population',''), tags=('population'))
        elif k==P: #and k<P+C
            fields_gr['table'].insert("", "end", values=('', 'Crossover',''), tags=('crossover'))
        elif k==P+2*C: #and k<P+C+M
            fields_gr['table'].insert("", "end", values=('', 'Mutation',''), tags=('mutation'))
        elif k==P+2*C+M:
            fields_gr['table'].insert("", "end", values=('', 'D',''), tags=('d'))
        fields_gr['table'].insert("", "end", values=(i[0], i[1],i[2]))
        k+=1

    fields_gr['table'].tag_configure('population', background='lightgreen')
    fields_gr['table'].tag_configure('crossover', background='blue')
    fields_gr['table'].tag_configure('mutation', background='orange')
    fields_gr['table'].tag_configure('d', background='grey')

fields={}
fields_gr={}

def pre_start ():
    global P, C, M, D
    '''P=int(fields['population'].get())
    C=int(fields['crossover'].get())
    M=int(fields['mutation'].get())
    D=int(fields['d'].get())'''

    global population, cross, mut, r_d
    population, cross, mut, r_d = [],[],[],[]
    global graphic_points, graphic_crossover,graphic_mutation,grahp_elements,graphic_r_d
    graphic_points, graphic_crossover,graphic_mutation,grahp_elements,graphic_r_d=[],[],[],[],[]

    algor(0)
    grahp()
    update_canvas()
    #main()

def re_pop():    
    global cross, mut, r_d
    cross, mut, r_d = [],[],[]
    global graphic_points, graphic_crossover,graphic_mutation,grahp_elements,graphic_r_d
    graphic_points, graphic_crossover,graphic_mutation,grahp_elements,graphic_r_d=[],[],[],[],[]
    algor(1)
    grahp()
    update_canvas()



def main():
    root = Tk()
    root.geometry('1600x600')

    #Canva
    fig = plt.gcf()
    fields_gr['canvas'] = FigureCanvasTkAgg(fig, root)
    canvas_widget = fields_gr['canvas'].get_tk_widget()
    canvas_widget.pack(side='right',fill=Y, padx=15, pady=30)
    
    
    '''fig = plt.gcf()

    fields_gr['canvas'] = FigureCanvasTkAgg(fig, root)
    canvas_widget = fields_gr['canvas'].get_tk_widget()
    canvas_widget.pack(side='right')'''

    #TABLE
    global grahp_elements

    fields_gr['table'] = ttk.Treeview(root, columns=('X','Y','F'), show="headings")
    fields_gr['table'].heading("X", text="X")
    fields_gr['table'].heading("Y", text="Y")
    fields_gr['table'].heading("F", text="F")
    for i in grahp_elements:
        for x, y, f in zip(i[0], i[1],i[2]):
            fields_gr['table'].insert("", "end", values=(x, y,f))

    fields_gr['table'].pack(side='right',fill=Y, padx=15,pady=30)

    #Entry
    
    fields['population_label'] = ttk.Label(text='Population:')
    fields['population'] = ttk.Entry()

    fields['crossover_label'] = ttk.Label(text='Crossover:')
    fields['crossover'] = ttk.Entry()

    fields['mutation_label'] = ttk.Label(text='Mutation:')
    fields['mutation'] = ttk.Entry()

    fields['d_label'] = ttk.Label(text='D:')
    fields['d'] = ttk.Entry() #fields['d'] = ttk.Entry(show="*")
    
    for field in fields.values():
        field.pack(anchor=tk.W, padx=20, pady=5, fill=None)
        
    ttk.Button(text='P.P', command=pre_start).pack(anchor=tk.W, padx=45, pady=25)
    ttk.Button(text='ReP', command=re_pop).pack(anchor=tk.W, padx=45, pady=3)
        
    '''population_label = ttk.Label(root, text="Population:").pack(side=tk.LEFT, padx=5)
    name_entry = ttk.Entry(root).pack(side=tk.LEFT, expand=False, fill=tk.X, padx=5)

    crossover_label = ttk.Label(root, text="Crossover:").pack(side=tk.LEFT, padx=5)
    crossover_entry = ttk.Entry(root).pack(side=tk.LEFT, expand=False, fill=tk.X, padx=5)
    
    mutation_label = ttk.Label(root, text="Mutation:").pack(side=tk.LEFT, padx=5)
    mutation_entry = ttk.Entry(root).pack(side=tk.LEFT, expand=False, fill=tk.X, padx=5)
    
    d_label = ttk.Label(root, text="D:").pack(side=tk.LEFT, padx=5)
    d_entry = ttk.Entry(root).pack(side=tk.LEFT, expand=False, fill=tk.X, padx=5)
    
    
    Label(root, text='Population').grid(row=0)
    Label(root, text='Crossover').grid(row=1)
    Label(root, text='Mutation').grid(row=2)
    Label(root, text='D').grid(row=3)
    e1.grid(row=0, column=1)
    e2.grid(row=1, column=1)
    e3.grid(row=2, column=1)
    e4.grid(row=3, column=1)

    pp_button = Button(root, text="P.P", command=algor)
    pp_button.pack(side=tk.TOP, expand=True,fill=tk.NONE)'''


    mainloop()


main()
'''algor()
prepare_for_graphic()
grahp()'''