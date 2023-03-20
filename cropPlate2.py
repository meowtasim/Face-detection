import matplotlib.pyplot as plt
import cv2 
numberPlate_cascade = "D:\meowtwo\Sem4\slot15\pl.xml"
detector = cv2.CascadeClassifier(numberPlate_cascade)
img = cv2.imread('D:\meowtwo\Sem4\slot15\images\plate7.jfif')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plates = detector.detectMultiScale(
       img_gray,scaleFactor=1.05,minNeighbors=7,
      minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
print(plates)
for (x,y,w,h) in plates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    plateROI = img_gray[y:y+h,x:x+w]
resize_test_license_plate = cv2.resize(
  plateROI, None, fx = 5, fy = 5,
  interpolation = cv2.INTER_CUBIC)
gaussian_blur_license_plate = cv2.GaussianBlur(
  resize_test_license_plate, (5, 5), 0)
plt.imshow(gaussian_blur_license_plate,cmap="gray")
# cv2.imshow("Cropped Image", gaussian_blur_license_plate)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.show()