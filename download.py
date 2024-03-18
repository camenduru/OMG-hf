import os
import time
import torch
#os.environ['HF_ENDPOINT']="https://hf-mirror.com"
from huggingface_hub import hf_hub_download

class OMG_download():
    def __init__(self) -> None:
        self.download_model_sam()
        print("download efficientvit sam")

        self.download_model_yoloworld()
        print("download yoloworld")

        self.download_controlNet()
        print("download controlNet")

        self.download_characters()
        print("download character")

        self.download_styles()
        print("download styles")


    def download_model_sam(self):
        REPO_ID = 'han-cai/efficientvit-sam'
        filename_list = ['xl1.pt']
        if not os.path.exists('/home/user/app/checkpoint/sam/'):
            os.makedirs('/home/user/app/checkpoint/sam/')
        for filename in filename_list:
            local_file = os.path.join('/home/user/app/checkpoint/sam/', filename)

            if not os.path.exists(local_file):
                hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir='/home/user/app/checkpoint/sam/', local_dir_use_symlinks=False)

    def download_model_yoloworld(self):
        REPO_ID = 'Fucius/OMG'
        filename_list = ['yolo-world.pt']
        if not os.path.exists('/tmp/cache/yolo_world/l/'):
            os.makedirs('/tmp/cache/yolo_world/l/')
        for filename in filename_list:
            local_file = os.path.join('/tmp/cache/yolo_world/l/', filename)
            if not os.path.exists(local_file):
                hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir='/tmp/cache/yolo_world/l/', local_dir_use_symlinks=False)

    def download_controlNet(self):
        REPO_ID = 'lllyasviel/ControlNet'
        filename_list = ['annotator/ckpts/body_pose_model.pth']
        if not os.path.exists('/home/user/app/checkpoint/ControlNet/'):
            os.makedirs('/home/user/app/checkpoint/ControlNet/')
        for filename in filename_list:
            local_file = os.path.join('/home/user/app/checkpoint/ControlNet/', filename)

            if not os.path.exists(local_file):
                hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir='/home/user/app/checkpoint/ControlNet/',
                                local_dir_use_symlinks=False)

    def download_characters(self):
        REPO_ID = 'Fucius/OMG'
        filename_list = ['lora/chris-evans.safetensors', 'lora/Harry_Potter.safetensors', 'lora/Hermione_Granger.safetensors', 'lora/jordan_torres_v2_xl.safetensors', 'lora/keira_lora_sdxl_v1-000008.safetensors', 'lora/lawrence_dh128_v1-step00012000.safetensors', 'lora/Gleb-Savchenko_Liam-Hemsworth.safetensors', 'lora/TaylorSwiftSDXL.safetensors']
        if not os.path.exists('./checkpoints/'):
            os.makedirs('./checkpoints/')
        for filename in filename_list:
            local_file = os.path.join('./checkpoints/', filename)

            if not os.path.exists(local_file):
                hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir='./checkpoints/',
                                local_dir_use_symlinks=False)
    def download_styles(self):
        REPO_ID = 'Fucius/OMG'
        filename_list = ['style/EldritchPaletteKnife.safetensors', 'style/Cinematic Hollywood Film.safetensors', 'style/Anime_Sketch_SDXL.safetensors']
        if not os.path.exists('./checkpoints/'):
            os.makedirs('./checkpoints/')
        for filename in filename_list:
            local_file = os.path.join('./checkpoints/', filename)

            if not os.path.exists(local_file):
                hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir='./checkpoints/',
                                local_dir_use_symlinks=False)

if __name__ == '__main__':
    down = OMG_download()
    print("finished download")