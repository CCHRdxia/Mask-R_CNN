from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2


root1 = Tk()
frame = Frame(root1, bd=2, relief=SUNKEN)
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)
xscroll = Scrollbar(frame, orient=HORIZONTAL)
xscroll.grid(row=1, column=0, sticky=E+W)
yscroll = Scrollbar(frame)
yscroll.grid(row=0, column=1, sticky=N+S)
canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
canvas.grid(row=0, column=0, sticky=N+S+E+W)
xscroll.config(command=canvas.xview)
yscroll.config(command=canvas.yview)
frame.pack(fill=BOTH,expand=1)

i = 0
#function to be called when mouse is clicked
def printcoords():
    global i
    File = filedialog.askopenfilename(parent=root1, initialdir="F:/",title='Choose an image.')
    filename = ImageTk.PhotoImage(Image.open(File))
    canvas.image = filename  # <--- keep reference of your image
    canvas.create_image(0,0,anchor='nw',image=filename)
    if(i == 0):
        src = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/bb.png")
        src1 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/bbb.png")
        src2 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/bbbbb.png")
        cv2.imshow('image', src)
        cv2.imshow('image1', src1)
        cv2.imshow('image2', src2)
        cv2.waitKey(0)
    if (i == 1):
        src = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/cc.png")
        src1 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/ccc.png")
        src2 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/ccccc.png")
        cv2.imshow('image', src)
        cv2.imshow('image1', src1)
        cv2.imshow('image2', src2)
        cv2.waitKey(0)
    if (i == 2):
        src = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/hh.png")
        src1 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/hhh.png")
        src2 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/hhhhh.png")
        cv2.imshow('image', src)
        cv2.imshow('image1', src1)
        cv2.imshow('image2', src2)
        cv2.waitKey(0)

    i += 1

Button(root1,text='choose',command=printcoords).pack()
root1.mainloop()
