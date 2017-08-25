import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
class Deeplab(nn.Module):
    def __init__(self, size=(241,121)):
        super(Deeplab, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.fc6_1 = nn.Conv2d(512, 1024, 3, padding=6, dilation=6)
        self.fc7_1 = nn.Conv2d(1024, 1024, 1)
        self.fc8_1 = nn.Conv2d(1024, 12, 1)

        self.fc6_2 = nn.Conv2d(512, 1024, 3, padding=12, dilation=12)
        self.fc7_2 = nn.Conv2d(1024, 1024, 1)
        self.fc8_2 = nn.Conv2d(1024, 12, 1)

        self.fc6_3 = nn.Conv2d(512, 1024, 3, padding=18, dilation=18)
        self.fc7_3 = nn.Conv2d(1024, 1024, 1)
        self.fc8_3 = nn.Conv2d(1024, 12, 1)

        self.fc6_4 = nn.Conv2d(512, 1024, 3, padding=24, dilation=24)
        self.fc7_4 = nn.Conv2d(1024, 1024, 1)
        self.fc8_4 = nn.Conv2d(1024, 12, 1)

        #self.fc8_interp = nn.Upsample(scale_factor=8,mode='bilinear')
        self.fc8_interp = nn.Upsample(size=size, mode='bilinear')

    def weights_init(self, pretrained_dict={}):
        init.normal(self.fc8_1.weight.data, mean=0, std=0.01)
        init.constant(self.fc8_1.bias.data, 0)
        init.normal(self.fc8_2.weight.data, mean=0, std=0.01)
        init.constant(self.fc8_2.bias.data, 0)
        init.normal(self.fc8_3.weight.data, mean=0, std=0.01)
        init.constant(self.fc8_3.bias.data, 0)
        init.normal(self.fc8_4.weight.data, mean=0, std=0.01)
        init.constant(self.fc8_4.bias.data, 0)

        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        x = nn.ReLU()(self.conv1_1(x))
        x = self.pool1(F.relu(self.conv1_2(x)))
        x = nn.ReLU()(self.conv2_1(x))
        x = self.pool2(F.relu(self.conv2_2(x)))
        x = nn.ReLU()(self.conv3_1(x))
        x = nn.ReLU()(self.conv3_2(x))
        x = self.pool3(F.relu(self.conv3_3(x)))
        x = nn.ReLU()(self.conv4_1(x))
        x = nn.ReLU()(self.conv4_2(x))
        x = self.pool4(F.relu(self.conv4_3(x)))
        x = nn.ReLU()(self.conv5_1(x))
        x = nn.ReLU()(self.conv5_2(x))
        x = self.pool5(F.relu(self.conv5_3(x)))

        x1 = nn.Dropout2d(0.5)(nn.ReLU()(self.fc6_1(x)))
        x1 = nn.Dropout2d(0.5)(nn.ReLU()(self.fc7_1(x1)))
        x1 = self.fc8_1(x1)

        x2 = nn.Dropout2d(0.5)(nn.ReLU()(self.fc6_2(x)))
        x2 = nn.Dropout2d(0.5)(nn.ReLU()(self.fc7_2(x2)))
        x2 = self.fc8_2(x2)

        x3 = nn.Dropout2d(0.5)(nn.ReLU()(self.fc6_3(x)))
        x3 = nn.Dropout2d(0.5)(nn.ReLU()(self.fc7_3(x3)))
        x3 = self.fc8_3(x3)

        x4 = nn.Dropout2d(0.5)(nn.ReLU()(self.fc6_4(x)))
        x4 = nn.Dropout2d(0.5)(nn.ReLU()(self.fc7_4(x4)))
        x4 = self.fc8_4(x4)
        x = self.fc8_interp(x1 + x2 + x3 + x4)
        return x

    # def val(self, x):
    #     """test
    #         remove Dropout
    #     """
    #     x = F.relu(self.conv1_1(x))
    #     x = self.pool1(F.relu(self.conv1_2(x)))
    #     x = F.relu(self.conv2_1(x))
    #     x = self.pool2(F.relu(self.conv2_2(x)))
    #     x = F.relu(self.conv3_1(x))
    #     x = F.relu(self.conv3_2(x))
    #     x = self.pool3(F.relu(self.conv3_3(x)))
    #     x = F.relu(self.conv4_1(x))
    #     x = F.relu(self.conv4_2(x))
    #     x = self.pool4(F.relu(self.conv4_3(x)))
    #     x = F.relu(self.conv5_1(x))
    #     x = F.relu(self.conv5_2(x))
    #     x = self.pool5(F.relu(self.conv5_3(x)))
    #     # print x.size()
    #     x1 = F.relu(self.fc6_1(x))
    #     x1 = F.relu(self.fc7_1(x1))
    #     x1 = self.fc8_1(x1)
    #
    #     x2 = F.relu(self.fc6_2(x))
    #     x2 = F.relu(self.fc7_2(x2))
    #     x2 = self.fc8_2(x2)
    #
    #     x3 = F.relu(self.fc6_3(x))
    #     x3 = F.relu(self.fc7_3(x3))
    #     x3 = self.fc8_3(x3)
    #
    #     x4 = F.relu(self.fc6_4(x))
    #     x4 = F.relu(self.fc7_4(x4))
    #     x4 = self.fc8_4(x4)
    #
    #     x = self.fc8_interp(x1 + x2 + x3 + x4)
    #     return x



