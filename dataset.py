import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset
import torchvision as tv

import cv2


class DataSet(Dataset):

    def __init__(self, file_list, image_channels, image_size, transforms=None,
                 loader=tv.datasets.folder.default_loader, is_train=True):
        self.file_list, self.labels = gen_data(file_list, is_train)
        self.image_channels = image_channels
        assert self.image_channels == 3
        self.image_size = image_size
        self.transforms = transforms
        self.loader = loader
        self.is_train = is_train

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        labels = self.labels[index]

        image = self.loader(file_name)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, labels


def gen_data(file_list, is_train=False):
    with open(file_list, 'r') as f:
        lines = f.readlines()

    filenames, labels = [], []
    for line in lines:
        line = line.strip('\t\n').split(' ')
        path = line[0]
        label = np.asarray(line[1:], dtype=np.long)
        filenames.append(path)
        labels.append(label)

    filenames = np.asarray(filenames, dtype=np.str)
    labels = np.asarray(labels, dtype=np.long)
    return (filenames, labels)


# if __name__ == '__main__':
#     file_list = 'data/train_data/list.txt'
#
#     data_transforms = tv.transforms.Compose([
#         tv.transforms.Resize((112, 112)),
#         tv.transforms.ToTensor()
#     ])
#
#     train_dataset = DataSet(file_list, 3, 112, transforms=data_transforms)
#     for i in range(len(train_dataset)):
#         image, landmarks, attributes = train_dataset[i]
#         # cv2.imshow('0', image)
#         # cv2.waitKey(0)
#         # image = np.asarray(image, dtype=np.float32)
#         print(image.dtype)
#         print(landmarks.dtype)
#         print(attributes.dtype)
#         exit(0)
#     # file_list = 'data/train_data/list.txt'
#     # filenames, landmarks, attributes = gen_data(file_list)
#     # for i in range(len(filenames)):
#     #     filename = filenames[i]
#     #     landmark = landmarks[i]
#     #     attribute = attributes[i]
#     #     print(attribute)
#     #     img = cv2.imread(filename)
#     #     h, w, _ = img.shape
#     #     landmark = landmark.reshape(-1, 2)*[h, w]
#     #     for (x, y) in landmark.astype(np.int32):
#     #         cv2.circle(img, (x, y), 1, (0, 0, 255))
#     #     cv2.imshow('0', img)
#     #     cv2.waitKey(0)
