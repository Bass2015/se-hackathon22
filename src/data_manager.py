from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

TRAIN_DATA = '../data/train.csv'
TEST_DATA = '../data/test.csv'
TRAIN_IMG = '../data/train_test_data/train/'

def split_data_frame(df, train_frac=0.8):
    """
    Splits a pandas dataframe in train and test sets.

    Parameters
    -----------------
    df: pd.DataFrame
        The DataFrame to split
    train_frac: float
        The train fraction of the data
    """
    train = df.sample(frac=train_frac).reset_index()
    test = df.drop(train.index).reset_index()
    return train, test

class ImageDataset(Dataset):
    """
    Class that holds the images of the dataset. 
    Used to load the test dataset. 
    """
    def __init__(self,df,data_folder,transform):
        self.df = df
        self.transform = transform
        self.img_folder = data_folder
        self.image_names = self.df[:]['example_path']
           
    def __len__(self):
        """ 
        Returns
        ------------
        How many elements the dataset holds.
        """
        return len(self.image_names)
    
    def __getitem__(self,index):
        """
        Gets an item from the dataset

        Parameters:
        ----------- 
        index: int
            The index of the wanted item

        Returns
        ------------
            A dict containing one key "image", that points to
            an image.
        """
        image=Image.open(self.img_folder+self.image_names.iloc[index])
        image=self.transform(image)
        sample = {'image': image}
        return sample

class LabeledDataset(Dataset):
    """
    Class that holds the images and labels of the dataset. 
    Used to load the test dataset. 
    """
    def __init__(self,df,data_folder,transform):
        self.df = df
        self.transform = transform
        self.img_folder = data_folder
        self.image_names = self.df[:]['example_path']
        self.labels = self.df[:]['label']
            
    def __len__(self):
        """ 
        Returns
        ------------
        How many elements the dataset holds.
        """
        return len(self.image_names)

    def __getitem__(self,index):
        """
        Gets an item from the dataset

        Parameters:
        ----------- 
        index: int
            The index of the wanted item

        Returns
        ------------
            A dict containing an image an its label.
        """
        image=Image.open(self.img_folder+self.image_names.iloc[index])
        image=self.transform(image)
        targets=self.labels[index]
        sample = {'image': image,'labels':targets}
        return sample

class LoaderFactory():
    """
    Class that builds Dataloaders from pandas dataframes.
    """
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
        """
        Build a torch.utils.DataLoader from a pandas DataFrame. Converts the images
        to tensors and normalizes them.
        If train is set to True, the images are also passed through other transformations.

        Parameters
        ----------
        img_folder: str
            The path where the images are located
        df: pd.DataFrame
            The dataframe where the labels and the paths to the different images are.
        train: bool
            If set to True, the images pass through more transforms.
        batch_size: int
            The size of the sampled batch when training
        """
        if train:
            dataset = LabeledDataset(df, img_folder, self.train_transform)
        else: 
            dataset = ImageDataset(df, img_folder, self.test_transform)       
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    



