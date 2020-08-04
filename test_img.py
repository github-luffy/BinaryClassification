import torch
from model2 import MobileNetV2, ONet, PNet, BlazeLandMark
import cv2
import numpy as np
import copy
import torchvision as tv
import os

from PIL import Image


test_data_transforms = tv.transforms.Compose([
        tv.transforms.Resize((64, 64)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

if __name__ == '__main__':
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ONet()
    # model = model2.MobileNetV2()
    # model = model2.PNet()
    # model = model2.BlazeLandMark()

    #model.load_state_dict(torch.load('./models2/ONet_20191206/model_499.pth'))
    model = torch.load('./models2/ONet_20191206/model_499.pth')
    model.cuda()
    model.eval()

    #directory_name = './test_img'
    directory_name = './TEST_IMG'
    for filename in os.listdir(directory_name):
        print(filename)
        img = cv2.imread(directory_name + "/" + filename)
        #img = cv2.imread('./test.jpeg')
        PIL_image = Image.fromarray(img) 
        print(PIL_image.mode)
        with torch.no_grad():
            test_image = test_data_transforms(PIL_image)
           
            outputs = model(test_image.unsqueeze(0).cuda())
            #print('before',outputs)
            #probability = torch.max(outputs, dim=1)[1].data.cpu().numpy().squeeze()
            _, probability = torch.max(outputs.data, dim=1)
            #print('after',probability)
            probability = probability.data.cpu().numpy().tolist()
            print('label is :',probability)