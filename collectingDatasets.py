import os
import cv2

name=input("Enter your name :")
output_dir=f"dataset/{name}"
os.makedirs(output_dir,exist_ok=True)

cap=cv2.VideoCapture(0)
count=0

while True:
    _,frame=cap.read()
    cv2.imshow("Capture Image -> Press 's' to save and 'q' to quit",frame)
    key=cv2.waitKey(1)
    if key==ord('s'):
        cv2.imwrite(f'{output_dir}/{name}_{count}.jpg',frame)
        print(f"Saved {name}_{count}.jpg")

        count+=1
    elif key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
