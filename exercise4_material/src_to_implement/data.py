from torch.utils.data import Dataset
import torch
from skimage.io import imread
from skimage.color import gray2rgb
import torchvision as tv
from torchvision import transforms
import zipfile

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):

    def __init__(self, data, mode):

        # mode can be either val or train
        self.mode = mode

        # define parameters
        self.data = data
        self._transform = tv.transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(train_mean, train_std)])

        self.zfile = zipfile.ZipFile('./images.zip', mode="r")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # load zip-file
        file = self.zfile.open(self.data['filename'][index])
        gray_img = imread(file)

        # transform gray-image into RGB
        image = self._transform(gray2rgb(gray_img))

        # store image and label as torch.tensor
        image = image.clone().detach()
        label = torch.tensor([self.data['crack'][index], self.data['inactive'][index]])

        return image, label


