import cv2

img = cv2.imread("D:\meowtwo\Sem4\slot15\images\plate.jfif")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = w / h
    if 1.9 < aspect_ratio < 5.0 and w > 100:
        plate = img[y:y+h, x:x+w]
        new_width = 1 * w
        new_height = 1 * h

        resized_img = cv2.resize(plate, (new_width, new_height))

        cv2.imshow("License Plate", resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
