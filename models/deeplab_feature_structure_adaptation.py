import os
import torch
from collections import OrderedDict
from .base_model import BaseModel
import networks
import itertools
from torch.autograd import Variable

def get_parameters(model, parameter_name):
    for name, param in model.named_parameters():
        if name in [parameter_name]:
            return param

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

def define_D(which_netD):
    if which_netD == 'FFC':
        return networks.FFC()
    elif which_netD == 'MultPathdilationNet':
        return networks.MultPathdilationNet()
    elif which_netD == 'SinglePathdilationSingleOutputNet':
        return networks.SinglePathdilationSingleOutputNet()
    elif which_netD == 'SinglePathdilationMultOutputNet':
        return networks.SinglePathdilationMultOutputNet()

def define_D_structure(which_netD_structure):
    if which_netD_structure == 'dcgan_D_multOut':
        return networks.dcgan_D_multOut()

class deeplabGan(BaseModel):
    def name(self):
        return 'deeplabGan'

    def initialize(self, args):
        BaseModel.initialize(self, args)
        self.nb = args['batch_size']
        sizeH, sizeW = args['fineSizeH'], args['fineSizeW']

        self.input_A = self.Tensor(self.nb, args['input_nc'], sizeH, sizeW)
        self.input_B = self.Tensor(self.nb, args['input_nc'], sizeH, sizeW)
        self.input_A_label = torch.cuda.LongTensor(self.nb, args['input_nc'], sizeH, sizeW)

        self.netG = networks.netG().cuda(device_id=args['device_ids'][0])
        self.netD = define_D(args['net_D']).cuda(device_id=args['device_ids'][0])
        self.netG_structure = networks.netG_structure().cuda(device_id=args['device_ids'][0])
        self.netD_structure = define_D_structure(args['net_D_structure']).cuda(device_id=args['device_ids'][0])

        self.deeplabPart1 = networks.DeeplabPool1().cuda(device_id=args['device_ids'][0])
        self.deeplabPart2 = networks.DeeplabPool12Conv5_1().cuda(device_id=args['device_ids'][0])
        self.deeplabPart3 = networks.DeeplabConv5_22Fc8_interp().cuda(device_id=args['device_ids'][0])

        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        self.netG_structure.apply(weights_init)
        self.netD_structure.apply(weights_init)
        pretrained_dict = torch.load(args['weigths_pool'] + '/' + args['pretrain_model'])
        self.deeplabPart1.weights_init(pretrained_dict=pretrained_dict)
        self.deeplabPart2.weights_init(pretrained_dict=pretrained_dict)
        self.deeplabPart3.weights_init(pretrained_dict=pretrained_dict)

        # define loss functions
        self.criterionCE = torch.nn.CrossEntropyLoss(size_average=False)
        self.criterionAdv = networks.Advloss(use_lsgan=args['use_lsgan'], tensor=self.Tensor)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=args['lr_gan'], betas=(args['beta1'], 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=args['lr_gan'], betas=(args['beta1'], 0.999))
        self.optimizer_G_S = torch.optim.Adam(self.netG_structure.parameters(),
                                            lr=args['lr_gan'], betas=(args['beta1'], 0.999))
        self.optimizer_D_S = torch.optim.Adam(self.netD_structure.parameters(),
                                            lr=args['lr_gan'], betas=(args['beta1'], 0.999))

        ignored_params = list(map(id, self.deeplabPart3.fc8_1.parameters()))
        ignored_params.extend(list(map(id, self.deeplabPart3.fc8_2.parameters())))
        ignored_params.extend(list(map(id, self.deeplabPart3.fc8_3.parameters())))
        ignored_params.extend(list(map(id, self.deeplabPart3.fc8_4.parameters())))
        base_params = filter(lambda p: id(p) not in ignored_params,
                             self.deeplabPart3.parameters())
        base_params = base_params +  filter(lambda p: True, self.deeplabPart1.parameters())
        base_params = base_params +  filter(lambda p: True, self.deeplabPart2.parameters())


        self.optimizer_P = torch.optim.SGD([{'params': base_params},
            {'params': get_parameters(self.deeplabPart3.fc8_1, 'weight'), 'lr': args['l_rate'] * 10},
            {'params': get_parameters(self.deeplabPart3.fc8_2, 'weight'), 'lr': args['l_rate'] * 10},
            {'params': get_parameters(self.deeplabPart3.fc8_3, 'weight'), 'lr': args['l_rate'] * 10},
            {'params': get_parameters(self.deeplabPart3.fc8_4, 'weight'), 'lr': args['l_rate'] * 10},
            {'params': get_parameters(self.deeplabPart3.fc8_1, 'bias'), 'lr': args['l_rate'] * 20},
            {'params': get_parameters(self.deeplabPart3.fc8_2, 'bias'), 'lr': args['l_rate'] * 20},
            {'params': get_parameters(self.deeplabPart3.fc8_3, 'bias'), 'lr': args['l_rate'] * 20},
            {'params': get_parameters(self.deeplabPart3.fc8_4, 'bias'), 'lr': args['l_rate'] * 20},
        ], lr=args['l_rate'], momentum=0.9, weight_decay=5e-4)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        networks.print_network(self.netD)
        networks.print_network(self.netG_structure)
        networks.print_network(self.netD_structure)
        networks.print_network(self.deeplabPart1)
        networks.print_network(self.deeplabPart2)
        networks.print_network(self.deeplabPart3)
        print('-----------------------------------------------')


    def set_input(self, input):
        self.input = input
        input_A = input['A']
        input_A_label = input['A_label']
        input_B = input['B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_label.resize_(input_A_label.size()).copy_(input_A_label)
        self.input_B.resize_(input_B.size()).copy_(input_B)

    def forward(self):
        self.A = Variable(self.input_A)
        self.A_label = Variable(self.input_A_label)
        self.B = Variable(self.input_B)

    # used in test time, no backprop
    def test(self, adaptation, input):
        if not adaptation :
            self.input_A.resize_(input.size()).copy_(input)
            self.A = Variable(self.input_A)
            self.output = self.deeplabPart3(self.deeplabPart2(self.deeplabPart1(self.A)))
        else:
            self.input_B.resize_(input.size()).copy_(input)
            self.B = Variable(self.input_B)
            f1 = self.deeplabPart1(self.B)
            f2 = self.deeplabPart2(f1) + self.netG(f1)
            self.output = self.deeplabPart3(f2) + self.netG_structure(f2)

    def get_image_paths(self):
        pass

    def backward_P(self):
        self.feature_A = self.deeplabPart2(self.deeplabPart1(self.A))
        self.feature_A_input =  Variable(self.feature_A.data)
        self.predic_A = self.deeplabPart3(self.feature_A)
        self.predic_A_input = Variable(self.predic_A.data)
        self.output = Variable(self.predic_A.data)

        self.loss_P = self.criterionCE(self.predic_A, self.A_label) / self.nb
        self.loss_P.backward()

    def backward_G(self):
        self.input_B_G = self.deeplabPart1(self.B)
        self.resdual = self.netG(self.input_B_G)
        self.feature_B = self.deeplabPart2(self.input_B_G) + self.resdual
        self.feature_B_input = Variable(self.feature_B.data)
        pred_fake = self.netD.forward(self.feature_B)

        self.loss_G = self.criterionAdv(pred_fake, True)
        self.loss_G.backward()

    def backward_D(self):
        pred_real = self.netD.forward(self.feature_A_input)

        loss_D_real = self.criterionAdv(pred_real, True)

        pred_fake = self.netD.forward(self.feature_B_input)
        loss_D_fake = self.criterionAdv(pred_fake, False)

        self.loss_D = (loss_D_real + loss_D_fake) * 0.5

        self.loss_D.backward()

    def backward_G_S(self):
        self.resdual_S = self.netG_structure(self.feature_B_input)
        self.predict_B = self.deeplabPart3(self.input_B_G) + self.resdual_S
        self.predict_B_input = Variable(self.predict_B.data)
        pred_fake = self.netD.forward(self.predict_B)

        self.loss_G_S = self.criterionAdv(pred_fake, True)
        self.loss_G_S.backward()

    def backward_D_S(self):
        pred_real = self.netD.forward(self.predic_A_input)

        loss_D_S_real = self.criterionAdv(pred_real, True)

        pred_fake = self.netD.forward(self.predict_B_input)
        loss_D_S_fake = self.criterionAdv(pred_fake, False)

        self.loss_D_S = (loss_D_S_real + loss_D_S_fake) * 0.5

        self.loss_D_S.backward()


    def optimize_parameters(self):
        # forward
        self.forward()
        # deeplab
        self.optimizer_P.zero_grad()
        self.backward_P()
        self.optimizer_P.step()
        # G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # G_S
        self.optimizer_G_S.zero_grad()
        self.backward_G_S()
        self.optimizer_G_S.step()
        # D_S
        self.optimizer_D_S.zero_grad()
        self.backward_D_S()
        self.optimizer_D_S.step()

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        self.save_network(self.netG, 'netG', label, self.gpu_ids)
        self.save_network(self.netD, 'netD', label, self.gpu_ids)
        self.save_network(self.netG_structure, 'netG_structure', label, self.gpu_ids)
        self.save_network(self.netD_structure, 'netD_structure', label, self.gpu_ids)
        self.save_network(self.deeplabPart1, 'deeplabPart1', label, self.gpu_ids)
        self.save_network(self.deeplabPart2, 'deeplabPart2', label, self.gpu_ids)
        self.save_network(self.deeplabPart3, 'deeplabPart3', label, self.gpu_ids)

    def update_learning_rate(self):
        pass

    def train(self):
        self.deeplabPart1.train()
        self.deeplabPart2.train()
        self.deeplabPart3.train()
        self.netG.train()
        self.netD.train()
        self.netG_structure.train()
        self.netD_structure.train()

    def eval(self):
        self.deeplabPart1.eval()
        self.deeplabPart2.eval()
        self.deeplabPart3.eval()
        self.netG_structure.eval()
        self.netD_structure.eval()