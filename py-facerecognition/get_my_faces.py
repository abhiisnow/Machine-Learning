import cv2
import os
import sys
import random
import time

output_dir = './my_faces'
size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

camera = cv2.VideoCapture(0)
print('Start capture in 3 sec!')
time.sleep(3)

index = 1
while True:
    if (index <= 1000):
        print('Being processed picture %s' % index)
        success, img = camera.read()
        cv2.imshow("frame", img)
        cv2.imwrite(output_dir+'/'+str(index)+'.jpg', img)
        index += 1
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            camera.release()
            cv2.destroyAllWindows()
            print("Quit!")
            break
    else:
        cv2.destroyAllWindows()
        print('Finished!')
        break
