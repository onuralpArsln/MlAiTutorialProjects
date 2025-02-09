from pathlib import Path
import cv2
import os 
import numpy as np
import random 


folder=Path("Images")
counter=0
modeTanh=True


for entry in folder.iterdir():
    for png in entry.iterdir():
        if random.randint(1,10)>3:
            continue
        img=cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(128,128))
        img=img/255.0 # normalize to 0,1 good for sigmoid
        if modeTanh:
             img = (img * 2) - 1 #scale to tanh 

        counter+=1
        if modeTanh:
            cv2.imwrite(f"database/{counter}.png", ((img + 1) * 127.5).astype(np.uint8))  # Convert back to [0, 255] for saving
        else:
            cv2.imwrite(f"database/{counter}.png", (img * 255).astype(np.uint8))

        
