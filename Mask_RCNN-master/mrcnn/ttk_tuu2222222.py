from tkinter import *
import time
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

root = Tk()
frame1 = Frame(root)
frame2 = Frame(root)
frame3 = Frame(root)
frame4 = Frame(root)
var = StringVar()
var.set('点击按钮加载模型')

textLabel = Label(frame1,
                  textvariable=var,
                  justify=LEFT,
                  )
textLabel.pack(side=RIGHT)

i = 0
def callback():
    var.set('正在加载,3秒后验证')
    time.sleep(2)

def callback1():
    root = Toplevel()
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E + W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N + S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N + S + E + W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH, expand=1)



    # function to be called when mouse is clicked
    def printcoords():
        global i
        File = filedialog.askopenfilename(parent=root, initialdir="F:/", title='Choose an image.')
        filename = ImageTk.PhotoImage(Image.open(File))
        canvas.image = filename  # <--- keep reference of your image
        canvas.create_image(0, 0, anchor='nw', image=filename)

        if(i == 0):
            src = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/bb.png")
            src1 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/bbb.png")
            src2 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/bbbbb.png")
            cv2.imshow('image', src)
            cv2.imshow('image1', src1)
            cv2.imshow('image2', src2)
            cv2.waitKey(0)
        if(i == 1):
            src = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/dd.png")
            src1 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/ddd.png")
            src2 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/ddddd.png")
            cv2.imshow('image', src)
            cv2.imshow('image1', src1)
            cv2.imshow('image2', src2)
            cv2.waitKey(0)
        if (i == 2):
            src = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/ee.png")
            src1 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/eee.png")
            src2 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/eeeee.png")
            cv2.imshow('image', src)
            cv2.imshow('image1', src1)
            cv2.imshow('image2', src2)
            cv2.waitKey(0)
        if (i == 3):
            src = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/ff.png")
            src1 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/fff.png")
            src2 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/fffff.png")
            cv2.imshow('image', src)
            cv2.imshow('image1', src1)
            cv2.imshow('image2', src2)
            cv2.waitKey(0)
        if (i == 4):
            src = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/hh.png")
            src1 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/hhh.png")
            src2 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/hhhhh.png")
            cv2.imshow('image', src)
            cv2.imshow('image1', src1)
            cv2.imshow('image2', src2)
            cv2.waitKey(0)
        if (i == 5):
            src = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/ii.png")
            src1 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/iii.png")
            src2 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/iiiii.png")
            cv2.imshow('image', src)
            cv2.imshow('image1', src1)
            cv2.imshow('image2', src2)
            cv2.waitKey(0)

        i += 1


    Button(root, text='choose', command=printcoords).pack()
    root.mainloop()


def callback2():
    var.set('加载完成，请选择指纹图片')
    time.sleep(1)

theButton = Button(frame2, text='加载模型', command=callback)
theButton.pack()
theButton = Button(frame4, text='验证模型', command=callback2)
theButton.pack()
theButton = Button(frame3, text='选择图片',command=callback1)
theButton.pack()

frame1.pack(padx=100, pady=60)
frame2.pack(padx=100, pady=60)
frame4.pack(padx=100, pady=60)
frame3.pack(padx=100, pady=60)

mainloop()