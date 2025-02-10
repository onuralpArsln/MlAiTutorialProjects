from pathlib import Path
import cv2
import os 
import numpy as np
import random 


folder=Path("dodTest")
counter=0
modeTanh=True


for entry in folder.iterdir():
    for png in entry.iterdir():
        if random.randint(1,10)>11:
            continue
        img=cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        img = 255 - img  # invert
        kernel = np.ones((3,3), np.uint8)  # Adjust kernel size as needed
        thickened = cv2.dilate(img, kernel, iterations=5) #thicken lines
        img = cv2.resize(thickened, (32, 32), interpolation=cv2.INTER_AREA)
        ret,thresh= cv2.threshold(img,100,255,cv2.THRESH_BINARY)
        counter+=1
        cv2.imwrite(f"dbtest/{counter}.png",thresh)  

        




