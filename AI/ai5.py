from collections import OrderedDict
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
import os

ABO=[]
NBO=[]
image=None
image_learn=[]
segments=None
n=3
threshold = 0.25 #поріг, у rgb2gray рідко зустрічається стого чорний
k=2
clf=None
image_learn_name=[]

def upload_image_for_learn():
    global image_learn
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
    
    learning()

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
            '''if image.ndim == 3:
                random_color = np.random.randint(0, 256, size=3, dtype=np.uint8)
                for c in range(3):
                    segm_image[..., c][mask] = random_color[c]
            else:'''
            random_gray = np.random.randint(0, 256, dtype=np.uint8)
            segm_image[mask] = random_gray

            '''
                if image.ndim == 3:
                for c in range(3):
                    try:
                        k = int(np.mean(image[..., c][mask]))
                    except:
                        k=0
                    segm_image[..., c][mask] = k
            else:
                segm_image[mask] = int(np.mean(image[mask]))'''
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
    global ABO, NBO, image_learn,threshold
    numb=1
    NBO.clear()
    ABO.clear()  
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
            print(' k  ',  np.sum(i[mask]))
            black_pixels.append(np.sum(i[mask] < threshold)) 

        '''k=1
        for i in black_pixels:
            ABO_part.append([k,black_pixels])
            k+=1
        print('abs_p ',ABO_part)

        k=1
        for i in black_pixels:
            NBO_part.append([k,i/max(black_pixels)])
            k+=1
        print('norm_p ',NBO_part)'''
        for j in black_pixels:
            NBO_part.append(j/max(black_pixels))

        ABO.append([numb,black_pixels])
        NBO.append([numb,NBO_part])
        print('\n\nabs ',ABO)
        print('\n\nnorms ',NBO)
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
    global ABO,NBO,clf
    
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

def learning():
    global threshold,image,image_learn
    
    X, y = [], []

    for i in NBO:
        X.append(i[1])
        y.append(i[0])
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0) #standartization

    w = np.random.randn(3) * 0.5
    b = 0.0
    eta = 0.1

    print("\n\nW ", w, "b ", b,"eta ", eta)
    
    epochs = 50
    for j in range(epochs):
        for i in range(len(X)):
            net = np.dot(X[i], w) + b
            y_pred = 1 if net >= 0 else 0
            error = y[i] - y_pred
            w += eta * error * X[i]
            b += eta * error
    print("After w ", w, "b ", b)

    
    outputs = np.dot(X, w) + b
    preds = (outputs >= 0).astype(int)


    X_new = np.array(img_check(), dtype=float)
    print("\nX_new ",X_new)
    X_new = (X_new - np.mean(X, axis=0)) / np.std(X, axis=0)
    result = np.dot(X_new, w) + b
    class_pred = 7 if result >= 0 else 5

    
    print("Res ", result)
    print("class_pred ", class_pred)


    result_label.config(text = f"Classified as {class_pred}")


def img_check():
    global clf, k, image
    h, w = image.shape

    try:
        k=int(fields['segments1'].get())
    except:
        print('\n     input K!\n')

    Y, X = np.mgrid[0:h, 0:w]
    angles = np.arctan2(Y, X)
    angles = (angles - angles.min()) / (angles.max() - angles.min())
    segments = (angles * n).astype(int)
    
    black_pixels = []
    for i in range(n):
        mask = segments == i
        #print(' i  ',  np.sum(image[mask]))
        black_pixels.append(np.sum(image[mask] < threshold)) 
    #print('\n\n b ',black_pixels)

    return black_pixels


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
  
ttk.Button(text='Teacher', command=learning).pack(anchor=tk.W, padx=45, pady=3)
ttk.Button(text='Check', command=img_check).pack(anchor=tk.W, padx=45, pady=3)
    
root.mainloop()