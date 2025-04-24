import lightning as L
import torch
import torch.nn as nn

from .loss import Loss


def pad(k, p):
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=None, d=1, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, pad(k, p), dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, momentum=0.03, eps=1e-3)
        self.act = nn.LeakyReLU(0.01, inplace=True) if act else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    @staticmethod
    def forward(x):
        return torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)



class Backbone(nn.Module):
    def __init__(self, num_classes=1000, init_weight=True):
        super().__init__()

        self.features = nn.Sequential(
            Conv(3, 64, 7, 2),
            nn.MaxPool2d(2, 2),
            
            Conv(64, 192, 3),
            nn.MaxPool2d(2, 2),
            
            Conv(192, 128, 1),
            Conv(128, 256, 3),
            Conv(256, 256, 1),
            Conv(256, 512, 3),
            nn.MaxPool2d(2, 2),
            
            Conv(512, 256, 1),
            Conv(256, 512, 3),
            Conv(512, 256, 1),
            Conv(256, 512, 3),
            Conv(512, 256, 1),
            Conv(256, 512, 3),
            Conv(512, 256, 1),
            Conv(256, 512, 3),
            Conv(512, 512, 1),
            Conv(512, 1024, 3),
            nn.MaxPool2d(2, 2),
            
            Conv(1024, 512, 1),
            Conv(512, 1024, 3),
            Conv(1024, 512, 1),
            Conv(512, 1024, 3)
        )

        self.classifier = nn.Sequential(
            *self.features,
            GlobalAvgPool2d(),
            nn.Linear(1024, num_classes)
        )

        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        return self.classifier(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class Head(nn.Module):
    def __init__(self, feature_size, num_boxes, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            Conv(1024, 1024, 3),
            Conv(1024, 1024, 3, 2),
            Conv(1024, 1024, 3),
            Conv(1024, 1024, 3)
        )

        self.detect = nn.Sequential(
            Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, feature_size * feature_size * (5 * num_boxes + num_classes)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.detect(x)
        return x


class YOLO(L.LightningModule):
    def __init__(self, num_classes=20, num_anchors=2, num_features=7, pretrained_backbone=False):
        super().__init__()
        
        self.num_features = num_features
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        if pretrained_backbone:
            darknet = Backbone()
            darknet = nn.DataParallel(darknet)
            src_state_dict = torch.load('model_best.pth.tar')['state_dict']
            dst_state_dict = darknet.state_dict()

            for key in dst_state_dict.keys():
                print(f'Loading weight of {key}')
                dst_state_dict[key] = src_state_dict[key]
            
            darknet.load_state_dict(dst_state_dict)
            self.features = darknet.module.features
        else:
            self.features = Backbone().features
            
        self.head = Head(self.num_features, self.num_anchors, self.num_classes)
        self.loss = Loss()

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        x = x.view(-1, self.num_features, self.num_features, 5 * self.num_anchors + self.num_classes)
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._compute_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._compute_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def _compute_loss(self, y_hat, y):
        return self.loss(y_hat, y)
