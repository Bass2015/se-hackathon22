from PIL import Image
from torchvision import transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

TRAIN_DATA = '../data/train.csv'
TEST_DATA = '../data/test.csv'
TRAIN_IMG = '../data/train_test_data/train/'

def split_data_frame(df, train_frac=0.8):
    train = df.sample(frac=train_frac).reset_index()
    test = df.drop(train.index).reset_index()
    return train, test

class ImageDataset(Dataset):
    def __init__(self,df,data_folder,transform):
        self.df = df
        self.transform = transform
        self.img_folder = data_folder
        self.image_names = self.df[:]['example_path']
           
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self,index):
        image=Image.open(self.img_folder+self.image_names.iloc[index])
        image=self.transform(image)
        sample = {'image': image}
        return sample

class LabeledDataset(Dataset):
    def __init__(self,df,data_folder,transform):
        self.df = df
        self.transform = transform
        self.img_folder = data_folder
        self.image_names = self.df[:]['example_path']
        self.labels = self.df[:]['label']
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self,index):
        image=Image.open(self.img_folder+self.image_names.iloc[index])
        image=self.transform(image)
        targets=self.labels[index]
        sample = {'image': image,'labels':targets}
        return sample

class LoaderFactory():
    def __init__(self):
        self.test_transform = transforms.Compose([
                       transforms.ToTensor(), 
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.train_transform = transforms.Compose([
                       transforms.RandomHorizontalFlip(p=1),
                       transforms.RandomVerticalFlip(p=1),
                       transforms.RandomRotation(degrees=0.66),
                       transforms.ToTensor(), 
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                       
    def get_data_loader(self, img_folder, df, train=True, batch_size=20):
        if train:
            dataset = LabeledDataset(df, img_folder, self.train_transform)
        else: 
            dataset = ImageDataset(df, img_folder, self.test_transform)       
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    



