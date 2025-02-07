from pathlib import Path
import cv2
import os 
import numpy as np

folder=Path("Images")
counter=0

for entry in folder.iterdir():
    for png in entry.iterdir():
        img=cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(128,128))
        img=img/255.0 # normalize
        counter+=1
        cv2.imwrite(f"database/{counter}.png", (img * 255).astype(np.uint8))

        
