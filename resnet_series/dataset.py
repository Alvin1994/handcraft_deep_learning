"""Oranize my dataset."""

import os
import torch
from skimage import io
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    """My dataset."""

    def __init__(self, x, y, patterns, transform=None):
        """Initialize cls.

        Arguments:
            x {list} -- list of imgs
            y {list} -- list of {event: bool}
            patterns {list} -- event patterns

        Keyword Arguments:
            transform {transforms.Compose} -- transforms.Compose.
        """
        self.x = x
        self.y = y
        self.patterns = patterns
        self.transform = transform

    def __len__(self):
        """Get len of MyDataSet().

        Returns:
            int -- len of (X)
        """
        return len(self.x)

    def __getitem__(self, idx):
        """Call func of cls[idx].

        Arguments:
            idx {int/tensor(int)} -- idx

        Returns:
            (tensor(img), torch.LongTensor) -- (  
                                                    single img,
                                                    onehot vector)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = os.path.abspath(self.x[idx])
        try:
            image = io.imread(path)
        except FileNotFoundError:
            print("{:} not exist!".format(path))
            return
        if self.transform is not None:
            image = self.transform(image)

        if self.y is None:
            onehotVector = []
        else:    
            onehotVector = self.onehot_encoding(self.y[idx])

        return (image, torch.LongTensor(onehotVector), path)
    
    def onehot_encoding(self, label):
        """Encode {event: bool} to one hot vector under self.patterns.

        Arguments:
            label {dict()} -- {event: bool}

        Returns:
            list -- onehot vector
        """
        rst = list()
        for pattern_ in self.patterns:
            rst.append(int(label[pattern_]))
        return rst
if __name__ == '__main__':
    datset = MyDataSet(1,1,1)