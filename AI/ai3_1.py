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


ABO=[]
NBO=[]
image=None
image_learn=[]
segments=None
n=5
threshold = 0.25 #поріг, у rgb2gray рідко зустрічається стого чорний
k=4
kmeans=None
image_learn_name=[]
y=None

def upload_image_for_learn():
    global image_learn
    image_learn=[]
    file_path = filedialog.askopenfilenames(filetypes=[("Image Files", "*.bmp;")])
    for i in file_path:
        image_learn.append(rgb2gray(io.imread(i)))
        image_learn_name.append(os.path.basename(i).replace('.bmp',''))
        
    update_canvas(image_learn,1)

def upload_image_for_check():
    global image
    file_path = filedialog.askopenfilename(filetypes=[("Image File", "*.bmp;")])
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

        for j in black_pixels:
            NBO_part.append(j/max(black_pixels))

        ABO.append([image_learn_name[numb],black_pixels])
        NBO.append([image_learn_name[numb],NBO_part])
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

def learning():
    global threshold,image,image_learn, k,kmeans,y
    X, y = [], []

    for i in ABO:
        for j in i[1]:
            X.append([j])
            y.append(i[0])
    kmeans = KMeans(n_clusters=len(image_learn), random_state=42)
    X=np.array(X)
    y=np.array(y)
    
    # мат сподівання
    centers = []
    for label in np.unique(y):
        class_vectors = X[y == label]
        center = np.mean(class_vectors, axis=0)
        centers.append(center)

    X= []
    y= np.unique(y).tolist()
    for i in centers:
        X.append(i)
    print('\n\nX ',X)
    kmeans.fit(X,y)
    
    result_label['text'] = "Teach"

def img_check():
    global kmeans, k, image,y
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
        black_pixels.append(np.sum(image[mask] < threshold)) 
    
    center = np.mean(black_pixels, axis=0) 
    value = np.array(center).reshape(1, -1)
    y=np.array(y)

    mapping = {old: new for old, new in zip(kmeans.labels_, y)}

    prediction = kmeans.predict(value)
    result = [mapping[label] for label in prediction]
    result_label.config(text = f"Segment {k} classified as {result}")


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