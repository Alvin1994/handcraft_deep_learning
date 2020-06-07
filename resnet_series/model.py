"""NN Model."""
import os
import sys
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
sys.path.append('./logs/')
import log as log


def show_and_logging(logger, text, type='info'):
    """Print and content to logger.

    Arguments:
        logger {logging} -- logger handler
        text {str} -- content

    Keyword Arguments:
        type {str} -- type of saving contetn. (default: {'info'})

    Returns:
        Bool -- True if operation success.
    """
    assert isinstance(text, str)
    if type == 'info':
        logger.info(text)
        print(text)
    return True


class MyModel(nn.Module):
    """Resnet + FCN (onehot-encoding)."""

    def __init__(self, patterns):
        """Initialize cls.

        Arguments:
            patterns {list[str]} -- Patterns list.
        """
        super(MyModel, self).__init__()
        self.patterns = patterns

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.backbone = models.resnet18(pretrained=True)
        self.backbone = self.backbone.to(self.device)

        nums_ftrs = self.backbone.fc.out_features
        self.fcs = nn.ModuleList([
            nn.Linear(nums_ftrs, 2).to(self.device) for _ in range(len(self.patterns))
        ])

        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(
        #     self.backbone.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.SGD(
            self.parameters(), lr=0.001, momentum=0.9)
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1
        )

        self.log = log.loggerOperation.init('./logs/model.log')

    def show_moel_info(self):
        """Show model structure.

        Returns:
            bool -- is operation success.
        """
        return print(self.parameters)

    def forward(self, x):
        """Foward func of nn.

        Arguments:
            x {torch.FloatTensor} -- input x.

        Returns:
            List[torch.FloatTensor, torch.cuda.FloatTensor] 
            -- output y and x.
        """
        x = x.to(self.device)
        backbone_output = self.backbone(x)

        y = list()
        for cuzFc in self.fcs:
            y.append(cuzFc(backbone_output))

        return [y, x]

    def load_model(self, path):
        """Load Model parameters in tensor.dict type.

        Arguments:
            path {str} -- Path to your parameters of model.
        """
        self.load_state_dict(torch.load(path))

    def prediction(self, dataLoader):
        """Predict outcome according to input(dataloader).

        Arguments:
            dataLoader {torch.utils.data.DataLoader} -- dataset.

        Returns:
            dict{imgName:'issueTag':{event:Bool}} -- {imgName:'issueTag':{cls:Bool}}
        """
        self.eval()

        rst = dict()
        with torch.set_grad_enabled(False):
            # prediction based on batch size.
            for inputs, label, imgsPath in dataLoader:
                inputs = inputs.to(self.device)
                backbone_output = self.backbone(inputs)

                for i, cuzFc in enumerate(self.fcs):
                    outputs = cuzFc(backbone_output)
                    _, preds = torch.max(outputs, 1)

                    for i_path, i_preds in zip(imgsPath, preds):
                        node_, name_ = os.path.split(i_path)
                        if i_path not in rst:
                            rst[i_path] = {'issueTag': dict()}
                        rst[i_path]['issueTag'][self.patterns[i]] = bool(i_preds)
        return rst

    def trainMyModel(self, num_epochs, dataloaders):
        """Train Process. Beaware that model.eval() and 
        model.train() only effect BN and dropout layer.

        Arguments:
            num_epochs {int} -- number of epochs.
            dataloaders {torch.utils.data.DataLoader} 
                -- dataset.

        Returns:
            Bool -- True if operation success.
        """
        since = time.time()

        best_wts = copy.deepcopy(self.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('  -' * 10)

            # Calc acc of train and validate set in each epoch
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.train()
                else:
                    self.eval()

                running_loss = 0.0
                num_label = len(dataloaders['train'].dataset[0][1])
                running_corrects = [0 for _ in range(num_label)]

                # Iterate over data
                for inputs, labels, _ in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero gradients parameter
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        backbone_output = self.backbone(inputs)

                        # calc loss iterate batchs.
                        loss = 0.0
                        for i, cuzFc in enumerate(self.fcs):
                            outputs = cuzFc(backbone_output)
                            labels_ = torch.LongTensor(
                                [x[i].tolist() for x in labels]).to(self.device)

                            loss += self.criterion(outputs, labels_)
                            _, preds = torch.max(outputs, 1)
                            running_corrects[i] += torch.sum(
                                preds == labels_.data)

                        # backward & optimize
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # Sum up the loss
                    running_loss += loss.item() * inputs.size(0)

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = list()
                for i in range(len(running_corrects)):
                    epoch_acc.append(running_corrects[i].double(
                    ) / len(dataloaders[phase].dataset))
                    print("  {} {}".format(str(self.patterns[i]),
                                           str(epoch_acc[i].item())
                                           ))

                cuz_acc = torch.mean(torch.stack(epoch_acc))
                
                show_and_logging(
                    self.log,
                    '{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, cuz_acc)
                )

                # deep copy the best wts
                if phase == 'val' and cuz_acc > best_acc:
                    best_acc = cuz_acc
                    best_model_wts = copy.deepcopy(self.state_dict())

        time_elapsed = time.time() - since

        show_and_logging(
            self.log,
            'Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        show_and_logging(
            self.log,
            'Best val Acc: {:4f}'.format(best_acc)
        )

        # load best model weights
        self.load_state_dict(best_model_wts)
        return True


if __name__ == '__main__':
    test_ = ['aa', 'bb','ss']
    m = MyModel(test_)
    print(m.state_dict)
