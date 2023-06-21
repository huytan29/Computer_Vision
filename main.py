import numpy as np
import face_recognition
import os
from datetime import datetime
from simplecv import Camera, DrawingLayer, Image

os.environ['OPENCV_VIDEOIO_DEBUG'] = '1'

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = Image(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img_rgb = img.to_rgb()
        img_np = np.array(img_rgb.get_matrix())
        faceLocs = face_recognition.face_locations(img_np)
        if len(faceLocs) == 0:
            continue
        encode = face_recognition.face_encodings(img_np, faceLocs)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cam = Camera()

while True:
    img = cam.get_image()

    # Chuyển đổi hình ảnh thành đen trắng bằng SimpleCV
    gray = img.to_grayscale()

    imgS = gray.resize(0.25)

    facesCurFrame = face_recognition.face_locations(imgS.get_matrix())
    encodesCurFrame = face_recognition.face_encodings(imgS.get_matrix(), facesCurFrame)

    for faceLoc in facesCurFrame:
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        img.draw_rectangle((x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    img.show()

    if img.is_done() or img.keys_pressed():
        break
