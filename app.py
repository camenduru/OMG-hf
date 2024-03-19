import spaces
import sys
import os



os.system(f"git clone https://github.com/Curt-Park/yolo-world-with-efficientvit-sam.git")
cwd0 = os.getcwd()
cwd1 = os.path.join(cwd0, "yolo-world-with-efficientvit-sam")
os.chdir(cwd1)
os.system("make setup")
os.chdir(f"/home/user/app")

# os.system(f"pip install inference[yolo-world]==0.9.13")
# print("install inference-yolo-world")
os.system(f"python app2.py")