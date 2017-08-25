import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
###############################################################################
# Functions
###############################################################################
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class Advloss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(Advloss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


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
        self.dropout = nn.Dropout2d(0.5)
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
        x = self.pool1(nn.ReLU()(self.conv1_2(x)))
        x = nn.ReLU()(self.conv2_1(x))
        x = self.pool2(nn.ReLU()(self.conv2_2(x)))
        x = nn.ReLU()(self.conv3_1(x))
        x = nn.ReLU()(self.conv3_2(x))
        x = self.pool3(nn.ReLU()(self.conv3_3(x)))
        x = nn.ReLU()(self.conv4_1(x))
        x = nn.ReLU()(self.conv4_2(x))
        x = self.pool4(nn.ReLU()(self.conv4_3(x)))
        x = nn.ReLU()(self.conv5_1(x))
        x = nn.ReLU()(self.conv5_2(x))
        x = self.pool5(nn.ReLU()(self.conv5_3(x)))

        x1 = self.dropout(0.5)(nn.ReLU()(self.fc6_1(x)))
        x1 = self.dropout(0.5)(nn.ReLU()(self.fc7_1(x1)))
        x1 = self.fc8_1(x1)

        x2 = self.dropout(0.5)(nn.ReLU()(self.fc6_2(x)))
        x2 = self.dropout(0.5)(nn.ReLU()(self.fc7_2(x2)))
        x2 = self.fc8_2(x2)

        x3 = self.dropout(0.5)(nn.ReLU()(self.fc6_3(x)))
        x3 = self.dropout(0.5)(nn.ReLU()(self.fc7_3(x3)))
        x3 = self.fc8_3(x3)

        x4 = self.dropout(0.5)(nn.ReLU()(self.fc6_4(x)))
        x4 = self.dropout(0.5)(nn.ReLU()(self.fc7_4(x4)))
        x4 = self.fc8_4(x4)
        x = self.fc8_interp(x1 + x2 + x3 + x4)
        return x

class DeeplabPool1(nn.Module):
    def __init__(self, size=(241,121)):
        super(DeeplabPool1, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


    def weights_init(self, pretrained_dict={}):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        x = nn.ReLU()(self.conv1_1(x))
        x = self.pool1(nn.ReLU()(self.conv1_2(x)))
        return x

class DeeplabPool12Conv5_1(nn.Module):
    def __init__(self, size=(241,121)):
        super(DeeplabPool12Conv5_1, self).__init__()
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

        self.relu = nn.ReLU()

        #self.fc8_interp = nn.Upsample(scale_factor=8,mode='bilinear')
        self.fc8_interp = nn.Upsample(size=size, mode='bilinear')

    def weights_init(self, pretrained_dict={}):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        x = nn.ReLU()(self.conv2_1(x))
        x = self.pool2(nn.ReLU()(self.conv2_2(x)))
        x = nn.ReLU()(self.conv3_1(x))
        x = nn.ReLU()(self.conv3_2(x))
        x = self.pool3(nn.ReLU()(self.conv3_3(x)))
        x = nn.ReLU()(self.conv4_1(x))
        x = nn.ReLU()(self.conv4_2(x))
        x = self.pool4(nn.ReLU()(self.conv4_3(x)))
        x = nn.ReLU()(self.conv5_1(x))

        return x

class DeeplabConv5_22Fc8_interp(nn.Module):
    def __init__(self, size=(241,121)):
        super(DeeplabConv5_22Fc8_interp, self).__init__()
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
        self.relu = nn.ReLU()
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
        x = self.relu(self.conv5_2(x))
        x = self.pool5(self.relu(self.conv5_3(x)))

        x1 = nn.Dropout2d(0.5)(self.relu(self.fc6_1(x)))
        x1 = nn.Dropout2d(0.5)(self.relu(self.fc7_1(x1)))
        x1 = self.fc8_1(x1)

        x2 = nn.Dropout2d(0.5)(self.relu(self.fc6_2(x)))
        x2 = nn.Dropout2d(0.5)(self.relu(self.fc7_2(x2)))
        x2 = self.fc8_2(x2)

        x3 = nn.Dropout2d(0.5)(self.relu(self.fc6_3(x)))
        x3 = nn.Dropout2d(0.5)(self.relu(self.fc7_3(x3)))
        x3 = self.fc8_3(x3)

        x4 = nn.Dropout2d(0.5)(self.relu(self.fc6_4(x)))
        x4 = nn.Dropout2d(0.5)(self.relu(self.fc7_4(x4)))
        x4 = self.fc8_4(x4)
        x = self.fc8_interp(x1 + x2 + x3 + x4)
        return x


class netG(nn.Module):
    def __init__(self, n_blocks=6):
        super(netG, self).__init__()
        input_nc = 64
        ngf = 128
        norm_layer = nn.BatchNorm2d
        padding_type = 'reflect'
        use_dropout = 0

        mult = 1
        model = [nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1), norm_layer(ngf), nn.ReLU(True)]

        for i in range(n_blocks):
            if (i+1) % 3 == 0:
                model += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1), nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2,padding=1)]
                mult *= 2
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout)]

        self.model = nn.Sequential(*model)


    def forward(self, x):
        return self.model(x)

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__()
        input_nc = 512
        ngf = 128
        norm_layer = nn.BatchNorm2d
        padding_type = 'reflect'
        use_dropout = 0

        model_1 = [nn.Conv2d(512, 1024, 3, padding=6, dilation=6), nn.Dropout2d(0.5), nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                   nn.Dropout2d(0.5), nn.Conv2d(1024, 1, 3, stride=2, padding=1)]
        model_2 = [nn.Conv2d(512, 1024, 3, padding=12, dilation=12), nn.Dropout2d(0.5), nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                   nn.Dropout2d(0.5), nn.Conv2d(1024, 1, 3, stride=2, padding=1)]
        model_3 = [nn.Conv2d(512, 1024, 3, padding=18, dilation=18),nn.Dropout2d(0.5), nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                   nn.Dropout2d(0.5), nn.Conv2d(1024, 1, 3, stride=2, padding=1)]
        model_4 = [nn.Conv2d(512, 1024, 3, padding=24, dilation=24),nn.Dropout2d(0.5), nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                   nn.Dropout2d(0.5), nn.Conv2d(1024, 1, 3, stride=2, padding=1)]

        self.model_1 = nn.Sequential(*model_1)
        self.model_2 = nn.Sequential(*model_2)
        self.model_3 = nn.Sequential(*model_3)
        self.model_4 = nn.Sequential(*model_4)


    def forward(self, x):
        return self.model_1(x) + self.model_2(x) + self.model_3(x) + self.model_4(x)

class FFC(nn.Module):
    def __init__(self):
        super(netD, self).__init__()
        input_nc = 512
        ngf = 128
        norm_layer = nn.BatchNorm2d
        padding_type = 'reflect'
        use_dropout = 0

        model_1 = [nn.Conv2d(512, 1024, 3, padding=6, dilation=6), nn.Dropout2d(0.5), nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                   nn.Dropout2d(0.5), nn.Conv2d(1024, 1, 3, stride=2, padding=1)]
        model_2 = [nn.Conv2d(512, 1024, 3, padding=12, dilation=12), nn.Dropout2d(0.5), nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                   nn.Dropout2d(0.5), nn.Conv2d(1024, 1, 3, stride=2, padding=1)]
        model_3 = [nn.Conv2d(512, 1024, 3, padding=18, dilation=18),nn.Dropout2d(0.5), nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                   nn.Dropout2d(0.5), nn.Conv2d(1024, 1, 3, stride=2, padding=1)]
        model_4 = [nn.Conv2d(512, 1024, 3, padding=24, dilation=24),nn.Dropout2d(0.5), nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                   nn.Dropout2d(0.5), nn.Conv2d(1024, 1, 3, stride=2, padding=1)]

        self.model_1 = nn.Sequential(*model_1)
        self.model_2 = nn.Sequential(*model_2)
        self.model_3 = nn.Sequential(*model_3)
        self.model_4 = nn.Sequential(*model_4)


    def forward(self, x):
        return self.model_1(x) + self.model_2(x) + self.model_3(x) + self.model_4(x)