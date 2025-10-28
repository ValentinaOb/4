import math
import os
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from skimage.color import rgb2gray
import numpy as np
from tkinter import *
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import norm
from collections import OrderedDict


ABO=[]
NBO=[]
image=None
image_learn=[]
segments=None
n=3
threshold = 0.25 #поріг, у rgb2gray рідко зустрічається стого чорний
numb_instance=5
knn=None
image_learn_name=[]
centers=[]
V=[]
y_test=[]

centers=[]
r=0
x0=0
group,group1=None,None

def upload_image_for_learn():
    global image_learn,group,group1
    image_learn=[]
    #file_path = filedialog.askopenfilenames(filetypes=[("Image Files", "*.bmp;")])
    file_path =['C:/Users/user/Downloads/5.bmp', 'C:/Users/user/Downloads/5_1.bmp', 'C:/Users/user/Downloads/5_2.bmp',
                 'C:/Users/user/Downloads/7.bmp', 'C:/Users/user/Downloads/7_1.bmp', 'C:/Users/user/Downloads/7_2.bmp']
    k=0
    for i in file_path:
        image_learn.append(rgb2gray(io.imread(i)))
        if k==0:
            image_learn_name.append(os.path.basename(i).replace('.bmp',''))
        elif k==n-1:
            k=-1
        k+=1

    update_canvas(image_learn,1)

    #
    upload_image_for_check()
    segmentation()
    count_black_dots()
    
    find_x()
    learning(group,group1)
    x0=img_check()
    predict(x0)

def upload_image_for_check():
    global image
    #file_path = filedialog.askopenfilename(filetypes=[("Image File", "*.bmp;")])
    file_path ='C:/Users/user/Downloads/5.bmp'
    image = rgb2gray(io.imread(file_path))
        
    update_canvas(image,0)

def segmentation():
    global n, segments, image,image_learn
    try:
        n=int(fields['segments'].get())
    except:
        print('\n     input N!\n')
    
    if image is not None:
        h, w = image.shape[:2]
        Y, X = np.mgrid[0:h, 0:w]

        angles = np.arctan2(Y, X)
        angles = (angles - angles.min()) / (angles.max() - angles.min())  # [0;1]
        segments = (angles * n).astype(int)

        segm_image = np.zeros_like(image)

        for k in range(n):
            mask = segments == k
            random_gray = np.random.randint(0, 256, dtype=np.uint8)
            segm_image[mask] = random_gray

        update_canvas(segm_image,0)

    if image_learn is not None:
        s_images=[]
        for i in image_learn:
            h, w = i.shape[:2]
            Y, X = np.mgrid[0:h, 0:w]

            angles = np.arctan2(Y, X)
            angles = (angles - angles.min()) / (angles.max() - angles.min())  # [0;1]
            segments = (angles * n).astype(int)
            segm_image = np.zeros_like(i)

            for k in range(n):
                mask = segments == k
                random_gray = np.random.randint(0, 256, dtype=np.uint8)
                segm_image[mask] = random_gray
            s_images.append(segm_image)
        update_canvas(s_images,1)

def count_black_dots():
    global ABO, NBO, image_learn,threshold,image_learn_name
    numb=0
    NBO.clear()
    ABO.clear()  
    ran=0
    name=0
    for i in image_learn:
        NBO_part=[]    
        h, w = i.shape

        Y, X = np.mgrid[0:h, 0:w]
        angles = np.arctan2(Y, X)
        angles = (angles - angles.min()) / (angles.max() - angles.min())
        segments = (angles * n).astype(int)

        black_pixels = []
        for k in range(n):
            mask = segments == k
            black_pixels.append(np.sum(i[mask] < threshold)) 

        for j in black_pixels:
            NBO_part.append(j/max(black_pixels))

        if ran in range(0,3):
            ABO.append([image_learn_name[name],black_pixels])
            NBO.append([image_learn_name[name],NBO_part])
            ran+=1
        if ran==3:
            name+=1
            ran=0
        numb+=1
        
    update_tables()

def update_canvas(img, butt):  
    if butt==1:  
        fig1 =fields_gr['fig1']
        fig1.clear()
        for i, im in enumerate(img):
            ax1 = fig1.add_subplot(1, len(img), i+1)
            ax1.imshow(im, cmap="gray")
        fields_gr['canvas1'].draw()
    else:
        fig =fields_gr['fig']
        fig.clear()
        ax = fig.add_subplot(111)
        ax.imshow(img,cmap='gray')
        ax.axis("off")    
        fields_gr['canvas'].draw()

def update_tables():
    global ABO,NBO
    
    for item in fields_gr['table_ABO'].get_children():
        fields_gr['table_ABO'].delete(item)
    for i in ABO:            
        data = ' | '.join(map(str, i[1]))
        fields_gr['table_ABO'].insert("", "end", values=(i[0], data))

    for item in fields_gr['table_NBO'].get_children():
        fields_gr['table_NBO'].delete(item)
    for i in NBO:
        data = ' | '.join(map(str, i[1]))
        fields_gr['table_NBO'].insert("", "end", values=(i[0], data))
   
    result_label['text'] = "Teach"

def img_check():
    global knn, image,y,centers,V,y_test
    h, w = image.shape

    Y, X = np.mgrid[0:h, 0:w]
    angles = np.arctan2(Y, X)
    angles = (angles - angles.min()) / (angles.max() - angles.min())
    segments = (angles * n).astype(int)
    
    black_pixels = []
    for i in range(n):
        mask = segments == i
        black_pixels.append(np.sum(image[mask] < threshold)) 

    return black_pixels

def find_x():
    global ABO,group,group1,n
    y = list(OrderedDict.fromkeys(row[0] for row in ABO))
    group, group1=[],[]
    for label in y:
        filtered = [row[1] for row in ABO if row[0] == label]
        if y.index(label)==0:
            group.append(filtered)
        else:
            group1.append(filtered)

    M1 = np.mean(group, axis=0)
    M2 = np.mean(group1, axis=0)
    #S1 = np.std(group, axis=0, ddof=1)
    #S2 = np.std(group1, axis=0, ddof=1)

    print("\nM1 ", M1)
    print("M2 ", M2)

    x0 = (M1 + M2) / 2
    print("\nX0", x0)

    x_new=img_check()

    decision = np.where(x_new < x0, y[0], y[1])

    group_votes = np.sum(x_new < x0)
    group1_votes = n - group_votes

    predicted_group = y[0] if group_votes > group1_votes else y[1]

    print("\nNew object ", x_new)
    print("Decitions ", decision)
    print('Res ', predicted_group)
    result_label.config(text = f"Classified as {predicted_group}")

def learning(group, group1):
    global x0
    M1 = np.mean(group, axis=0)
    M2 = np.mean(group1, axis=0)
    x0 = (M1 + M2) / 2

def predict(x_new):
    global x0
    y = list(OrderedDict.fromkeys(row[0] for row in ABO))
    group_votes = np.sum(x_new < x0)
    group1_votes = len(x_new) - group_votes
    result=y[0] if group_votes > group1_votes else y[1]
    print('Res ', result)
    result_label.config(text = f"Classified as {result}")

fields={}
fields_gr={}
fields_butt={}

root = Tk()
root.geometry('1400x650')

    #Canva
fields_gr['fig'] = plt.figure(figsize=(4, 5))
fields_gr['canvas'] = FigureCanvasTkAgg(fields_gr['fig'], root)
fields_gr['canvas'].get_tk_widget().pack(side="right",fill='y', expand=True, padx=15, pady=30)

fields_gr['fig1'] = plt.figure(figsize=(8, 3))
fields_gr['canvas1'] = FigureCanvasTkAgg(fields_gr['fig1'], root)
fields_gr['canvas1'].get_tk_widget().pack(side="bottom", fill="both", expand=True, padx=15, pady=30)

    #TABLE
fields_gr['table_ABO'] = ttk.Treeview(root, columns=('Sgmt','Dots'), show="headings",height=20)
fields_gr['table_ABO'].heading("Sgmt", text="Sgmt")
fields_gr['table_ABO'].heading("Dots", text="Dots")
fields_gr['table_ABO'].column("Sgmt", width=30)
fields_gr['table_ABO'].column("Dots", width=180)
for i in ABO:
    for x, y in zip(i[0], i[1]):
        fields_gr['table_ABO'].insert("", "end", values=(x, y))
fields_gr['table_ABO'].pack(side='right', padx=15,pady=30)

fields_gr['table_NBO'] = ttk.Treeview(root, columns=('Sgmt','Dots'), show="headings",height=20)
fields_gr['table_NBO'].heading("Sgmt", text="Sgmt")
fields_gr['table_NBO'].heading("Dots", text="Dots")
fields_gr['table_NBO'].column("Sgmt", width=30)
fields_gr['table_NBO'].column("Dots", width=180)
for i in NBO:
    for x, y in zip(i[0], i[1]):
        fields_gr['table_NBO'].insert("", "end", values=(x, y))
fields_gr['table_NBO'].pack(side='right', padx=15,pady=30)

result_label = ttk.Label(root, text = '',font=("Arial", 14))
result_label.pack(padx=15, pady=15,side='bottom')
    #Entry    
fields['segments'] = ttk.Entry()
fields['segments1'] = ttk.Entry()
for field in fields.values():
    field.pack(anchor=tk.W, padx=10, pady=5, fill=None)
        
tk.Button(text="Upload Image Learn", command=upload_image_for_learn).pack(anchor=tk.W, padx=25, pady=15)
tk.Button(text="Upload Image Check", command=upload_image_for_check).pack(anchor=tk.W, padx=25, pady=15)

fields_butt['segm'] = ttk.Button(text='Segmentation', command=segmentation)
fields_butt['abo_nbo'] = ttk.Button(text='ABO&NBO', command=count_black_dots)
for fields_b in fields_butt.values():
    fields_b.pack(anchor=tk.W, padx=10, pady=5, side='left')    
  
ttk.Button(text='Find X', command=find_x).pack(anchor=tk.W, padx=45, pady=3)
ttk.Button(text='Check', command=img_check).pack(anchor=tk.W, padx=45, pady=3)
    
root.mainloop()