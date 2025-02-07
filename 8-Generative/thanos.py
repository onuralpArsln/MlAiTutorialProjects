import random
from pathlib import Path

folder= Path("database")

for i in folder.iterdir():
    if random.randint(0,10)>9:
        i.unlink()