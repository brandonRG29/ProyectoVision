import cv2
imagen = cv2.imread("test/photo.jpg")
cv2.imshow("ventana", imagen)
print ("Hola mundo")
cv2.waitKey(0)
