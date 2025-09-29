import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
import numpy as np
from tkinter import *
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog

ABO=[]
NBO=[]
image=None
gray_img=None
segments=None
n=0

def upload_image():
    global image,ABO,NBO
    ABO,NBO=[],[]
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.bmp; *.jpg")])
    if file_path:
        image = io.imread(file_path)

    update_canvas(image)

def to_grey_image():
    global image,gray_img
    
    gray_img = rgb2gray(image)

    update_canvas(gray_img)

def segmentation():
    global n, image,segments
    try:
        n=int(fields['segments'].get())
    except:
        print('\n     input N!\n')
    h, w = image.shape[:2]
    Y, X = np.mgrid[0:h, 0:w]

    angles = np.arctan2(Y, X)
    angles = (angles - angles.min()) / (angles.max() - angles.min())  # [0;1]
    segments = (angles * n).astype(int)

    segm_image = np.zeros_like(image)

    for k in range(n):
        mask = segments == k
        if image.ndim == 3:
            for c in range(3):
                try:
                    k = int(np.mean(image[..., c][mask]))
                except:
                    k=0
                segm_image[..., c][mask] = k
        else:
            segm_image[mask] = int(np.mean(image[mask]))
    
    update_canvas(segm_image)

def count_black_dots():
    global ABO, NBO
    
    segm_gray_image = np.zeros_like(gray_img)
    black_pixels=[]

    try:
        for k in range(n):
            mask = segments == k
            if gray_img.ndim != 3:
                black_pixels.append(np.sum(segm_gray_image[mask] == 0))
    except:
        print('Not Gray!')

    #sum_black_pxls=sum(black_pixels)

    k=1
    for i in black_pixels:
        ABO.append([k,black_pixels])
        k+=1
    print('abs ',ABO)

    k=1
    for i in black_pixels:
        NBO.append([k,i/max(black_pixels)])
        #NBO.append([k,i/sum_black_pxls])
        k+=1
    print('norm ',NBO)

    update_tables()

def update_canvas(img):    
    fig = plt.gcf()
    fig.clear() 
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    ax.axis("off")    
    fields_gr['canvas'].draw()

def update_tables():
    global ABO,NBO
    #ABO.sort(key=lambda row: row[1],reverse=True)
    #NBO.sort(key=lambda row: row[1],reverse=True)
    
    for item in fields_gr['table_ABO'].get_children():
        fields_gr['table_ABO'].delete(item)
    for i in ABO:
        fields_gr['table_ABO'].insert("", "end", values=(i[0], i[1]))


    for item in fields_gr['table_NBO'].get_children():
        fields_gr['table_NBO'].delete(item)
    for i in NBO:
        fields_gr['table_NBO'].insert("", "end", values=(i[0], i[1]))

fields={}
fields_gr={}

def main():
    root = Tk()
    root.geometry('1400x600')

    #Canva
    fig = plt.gcf()
    fields_gr['canvas'] = FigureCanvasTkAgg(fig, root)
    canvas_widget = fields_gr['canvas'].get_tk_widget()
    canvas_widget.pack(side='right', padx=15, pady=30)

    #TABLE
    global ABO, NBO

    fields_gr['table_ABO'] = ttk.Treeview(root, columns=('Sgmt','Dots'), show="headings",height=20)
    fields_gr['table_ABO'].heading("Sgmt", text="Sgmt")
    fields_gr['table_ABO'].heading("Dots", text="Dots")
    fields_gr['table_ABO'].column("Sgmt", width=50)
    fields_gr['table_ABO'].column("Dots", width=120)
    for i in ABO:
        for x, y in zip(i[0], i[1]):
            fields_gr['table_ABO'].insert("", "end", values=(x, y))
    fields_gr['table_ABO'].pack(side='right', padx=15,pady=30)

    fields_gr['table_NBO'] = ttk.Treeview(root, columns=('Sgmt','Dots'), show="headings",height=20)
    fields_gr['table_NBO'].heading("Sgmt", text="Sgmt")
    fields_gr['table_NBO'].heading("Dots", text="Dots")
    fields_gr['table_NBO'].column("Sgmt", width=50)
    fields_gr['table_NBO'].column("Dots", width=120)
    for i in NBO:
        for x, y in zip(i[0], i[1]):
            fields_gr['table_NBO'].insert("", "end", values=(x, y))
    fields_gr['table_NBO'].pack(side='right', padx=15,pady=30)


    #Entry    
    fields['segments'] = ttk.Label(text='Segments:')
    fields['segments'] = ttk.Entry()
    for field in fields.values():
        field.pack(anchor=tk.W, padx=20, pady=5, fill=None)

    for field in fields.values():
        field.pack(anchor=tk.W, padx=20, pady=5, fill=None)
        
    tk.Button(text="Upload Image", command=upload_image).pack(anchor=tk.W, padx=45, pady=25)
    ttk.Button(text='Image to B&W', command=to_grey_image).pack(anchor=tk.W, padx=45, pady=3)
    ttk.Button(text='Segmentation', command=segmentation).pack(anchor=tk.W, padx=45, pady=3)
    ttk.Button(text='ABO&NBO', command=count_black_dots).pack(anchor=tk.W, padx=45, pady=3)
    
    mainloop()

main()