
"""Train Script for nn model."""
import os
import glob
import json
import torch
import model as cuzModel
import dataset as cuzDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


def preprocess():
    """Process on data. 
        (1) load all jsons 
        (2). organizing data in {img, {event:bool}} 
        (3). split data in train and test set

    Returns:
        List() -- train set of X
        List() -- test  set of X
        List() -- train set of Y
        List() -- test  set of Y
        List() -- tuble of patterns
    """

    X, Y = list(), list()
    features = (
        'dog',
        'cat',
    )

    jsons_ = ['./imgs/label.json']

    for json_ in jsons_:
        node_, _ = os.path.split(json_)

        for key, val in read_json(json_).items():
            X.append(os.path.join(node_, key))
            Y.append(val)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.33,
        random_state=42)

    print('Preprocess')
    print('  Number of training set: {:}'.format(len(X_train)))
    print('  Number of testing set : {:}'.format(len(X_test)))
    print('  Features              : ' + str(features))
    return X_train, X_test, Y_train, Y_test, features


def read_json(file_loc):
    """Load json file.

    Arguments:
        file_loc {str} -- path to json

    Returns:
        dict() -- data
    """
    assert isinstance(file_loc, str)

    with open(file_loc) as f:
        data = json.load(f)
    f.close()
    return data


if __name__ == '__main__':

    X_train, X_test, Y_train, Y_test, patterns = preprocess()

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    datasets = {
        'train': cuzDataset.MyDataSet(X_train, Y_train, patterns, data_transforms['train']),
        'val': cuzDataset.MyDataSet(X_test, Y_test, patterns, data_transforms['val'])
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(datasets[x], batch_size=3, shuffle=True, num_workers=4) for x in ['train', 'val']
    }
 
    model = cuzModel.MyModel(patterns)
    model.show_moel_info()
    model.trainMyModel(20, dataloaders)
    # torch.save(model.state_dict(), 'model_manualAndGlass150_20.pkl')


