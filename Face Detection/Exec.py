import numpy as np
import cv2

'''
import urllib.request
import cv2
import numpy as np
import os

def store_raw_images():
    neg_images_link = '//image-net.org/api/text/imagenet.synset.geturls?wnid=n00523513'
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
    pic_num = 1

    if not os.path.exists('neg'):
        os.makedirs('neg')

    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, "neg/"+str(pic_num)+".jpg")
            img = cv2.imread("neg/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("neg/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1

        except Exception as e:
            print(str(e))

def find_uglies():
    match = False
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_image_path = str(file_type)+'/'+str(img)
                    ugly = cv2.imread('uglies/'+str(ugly))
                    question = cv2.imread(current_image_path)
                    if ugly.shape == question.shape and not(np.bitwise_xor(ugly,question).any()):
                        print('That is one ugly pic! Deleting!')
                        print(current_image_path)
                        os.remove(current_image_path)
                except Exception as e:
                    print(str(e))

def store_raw_images():
    neg_images_link = '//image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152'
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
    pic_num = 953

    if not os.path.exists('neg'):
        os.makedirs('neg')

    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, "neg/"+str(pic_num)+".jpg")
            img = cv2.imread("neg/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("neg/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1

        except Exception as e:
            print(str(e))

def create_pos_n_neg():
    for file_type in ['neg']:

        for img in os.listdir(file_type):

            if file_type == 'pos':
                line = file_type+'/'+img+' 1 0 0 50 50\n'
                with open('info.dat','a') as f:
                    f.write(line)
            elif file_type == 'neg':
                line = file_type+'/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)
'''


face_cascade = cv2.CascadeClassifier('C:\\Users\\Lakshya\\AppData\\Local\\Programs\\Python\\Python36-32\\Lib\\site-packages\\cv2\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\Lakshya\\AppData\\Local\\Programs\\Python\\Python36-32\\Lib\\site-packages\\cv2\\build\\etc\\haarcascades\\haarcascade_eye.xml')
watch_cascade = cv2.CascadeClassifier('C:\\Users\\Lakshya\\AppData\\Local\\Programs\\Python\\Python36-32\\Lib\\site-packages\\cv2\\build\\etc\\haarcascades\\myhaar.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    watches = watch_cascade.detectMultiScale(gray, 20, 20)


    for (x,y,w,h) in watches:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Watch',(x-w,y-h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Person',(x-w,y-h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
