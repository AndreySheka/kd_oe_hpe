import torch.nn
from torchvision import models


__all__ = ['Network']


class AnglesHead(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d((7, 7))
        self.pitch = torch.nn.Linear(size, 198)
        self.yaw = torch.nn.Linear(size, 198)
        self.roll = torch.nn.Linear(size, 198)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        pitch = self.pitch(x)
        yaw = self.yaw(x)
        roll = self.roll(x)

        return torch.stack([pitch, yaw, roll])


class Backbone(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.model = kind(pretrained=False)
        self.out_features = self.model.fc.in_features
        del self.model.fc
        del self.model.avgpool

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x


class Network(torch.nn.Module):
    def __init__(self, kind, state=None):
        super().__init__()
        self.encoder_net = Backbone(getattr(models, kind))
        self.head_net = AnglesHead(self.encoder_net.out_features)

        if state is not None:
            if isinstance(state, str):
                state = torch.load(state)

            self.load_state_dict(state)

        self.eval()

    def forward(self, x):
        x = self.encoder_net(x)
        return self.head_net(x)
