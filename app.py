import spaces
import sys
import os



os.system(f"git clone https://github.com/Curt-Park/yolo-world-with-efficientvit-sam.git")
cwd0 = os.getcwd()
cwd1 = os.path.join(cwd0, "yolo-world-with-efficientvit-sam")
os.chdir(cwd1)
os.system("make setup")
os.system(f"cd /home/user/app")
os.system("python app2.py")


