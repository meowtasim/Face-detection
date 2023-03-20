import cv2
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, storage
from firebase_admin import firestore
face_cascade = cv2.CascadeClassifier('o.xml')
db = firestore.client()
cap = cv2.VideoCapture(0)#Capture image from webcam
# cap = cv2.VideoCapture('filename.mp4')

#  Initialize Firebase app with credentials
cred = credentials.Certificate('D:\meowtwo\Sem4\slot15\hackathon\\firestore_service.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'safety-first-e6dc3.appspot.com'},{'projectId': 'safety-first-e6dc3'})#, {'storageBucket': 'your-storage-bucket.appspot.com'}

while True:
    fac=0
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (235, 255, 62), 2)
        cv2.putText(img, f"Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 255, 62), 2)
        fac+=1

    if fac>0:
        _, image_bytes = cv2.imencode('.jpg', img)
        image_bytes = image_bytes.tobytes()

        try:
            #upload
            bucket = storage.bucket()
            blob = bucket.blob('path/to/remote/image.jpg')
            blob.upload_from_string(image_bytes, content_type='image/jpeg')
            url = blob.generate_signed_url(expiration=300, method='GET')
            print('Downloadable URL:', url)
            currentTime = datetime.now()
            db.collection('facialRecognition').add({"facesDetected":url,"createdAt" : currentTime})
        except Exception as e:
            print('Error:', e)

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
cap.release()