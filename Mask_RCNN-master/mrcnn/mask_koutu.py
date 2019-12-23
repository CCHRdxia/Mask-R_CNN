import cv2

img1 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/i.jpg")
img2 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/iii.png")


#颜色空间转换，这里是BGR转到灰度空间，常用的还有BGR到HSV空间：cv2.COLOR_BGR2HSV
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# cv2.namedWindow("Image")
# cv2.imshow("Image",img2gray)
# cv2.waitKey(0)


#THRESH_BINARY=0，THRESH_BINARY_INV,THRESH_TRUNC,THRESH_TOZERO,THRESH_TOZERO_INV,THRESH_OTSU,THRESH_TRIANGLE,THRESH_MASK
#二值化函数threshold(src, thresh, maxval, type[, dst]),thresh:阈值，maxval：最大值，type：阈值类型
#ret:暂时就认为是设定的thresh阈值，mask：二值化的图像
ret,mask = cv2.threshold(img2gray,185,255,cv2.THRESH_BINARY)
# cv2.namedWindow("Image")
# cv2.imshow("Image", mask)
# cv2.imshow("Image", ret)
# cv2.waitKey(0)


cv2.imwrite("F:\PythonData\Mask_RCNN-master\mask_finger/iiii.png", mask)

img3 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/iiii.png")

image = cv2.add(img1, img3)
cv2.imwrite("F:\PythonData\Mask_RCNN-master\mask_finger/iiiii.png", image)


cv2.namedWindow("Image")
cv2.imshow("Image", image)
cv2.waitKey(0)