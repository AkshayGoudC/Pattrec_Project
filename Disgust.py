import os
import matplotlib.pyplot as plt
import cv2
from fer import FER

folder = r'C:\Semester-4\patt_Rec\Pattrec_Project\dataset\test\disgust'
disgust_count = 0
total_disgust_count = 0
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, filename))
    detector = FER(mtcnn=True)
    print(detector.detect_emotions(img))

    try:
        emotion,score = detector.top_emotion(img)
        if(emotion == 'disgust'):
            disgust_count+=1
        total_disgust_count+=1

    except :
        total_disgust_count+=1

result= (disgust_count/total_disgust_count)*100
print("Accuracy of Disgust dataset is ", result,"%")
