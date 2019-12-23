import cv2

src  = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/bb.png")
src1 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/bbb.png")
src2 = cv2.imread("F:\PythonData\Mask_RCNN-master\mask_finger/bbbbb.png")
cv2.imshow('image', src)
cv2.imshow('image1', src1)
cv2.imshow('image2', src2)
cv2.waitKey(0)