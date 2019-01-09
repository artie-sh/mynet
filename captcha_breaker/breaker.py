import cv2

myfile = cv2.imread("../captchas/847782.png")

imgray = cv2.cvtColor(myfile, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray,127,255,0)


im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for item in contours:
    print len(item)


