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
    elif which_netD == 'NoBNMultPathdilationNet':
        return networks.NoBNMultPathdilationNet()
    elif which_netD == 'SinglePathdilationSingleOutputNet':
        return networks.SinglePathdilationSingleOutputNet()
    elif which_netD == 'SinglePathdilationMultOutputNet':
        return networks.SinglePathdilationMultOutputNet()
    elif which_netD == 'NoBNSinglePathdilationMultOutputNet':
        return networks.NoBNSinglePathdilationMultOutputNet()

class deeplabGanWithRefine(BaseModel):
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

        self.deeplabPart1 = networks.DeeplabPool1().cuda(device_id=args['device_ids'][0])
        self.deeplabPart2 = networks.DeeplabPool12Pool5().cuda(device_id=args['device_ids'][0])
        self.deeplabPart3 = networks.DeeplabPool52Fc8_interp().cuda(device_id=args['device_ids'][0])

        # define loss functions
        self.criterionCE = torch.nn.CrossEntropyLoss(size_average=False)
        self.criterionAdv = networks.Advloss(use_lsgan=args['use_lsgan'], tensor=self.Tensor)


        if not args['resume']:
            #initialize networks
            self.netG.apply(weights_init)
            self.netD.apply(weights_init)
            pretrained_dict = torch.load(args['weigths_pool'] + '/' + args['pretrain_model'])
            self.deeplabPart1.weights_init(pretrained_dict=pretrained_dict)
            self.deeplabPart2.weights_init(pretrained_dict=pretrained_dict)
            self.deeplabPart3.weights_init(pretrained_dict=pretrained_dict)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=args['lr_gan'], betas=(args['beta1'], 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=args['lr_gan'], betas=(args['beta1'], 0.999))
        self.optimizer_R = torch.optim.SGD(self.netG.parameters(), lr=args['lr_refine'], momentum=0.9,
                                           weight_decay=5e-4)

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
            self.output = self.deeplabPart3(self.deeplabPart2(self.deeplabPart1(self.B)) + self.netG(self.deeplabPart1(self.B)))

    def get_image_paths(self):
        pass

    def backward_P(self):
        # Maintain pool5_B in this status
        self.pool5_B_input = Variable(self.deeplabPart2(self.deeplabPart1(self.B)).data)

        self.pool5_A = self.deeplabPart2(self.deeplabPart1(self.A))
        self.pool5_A_input =  Variable(self.feature_A.data)
        self.predic_A = self.deeplabPart3(self.feature_A)
        self.output = Variable(self.predic_A.data)

        self.loss_P = self.criterionCE(self.predic_A, self.A_label) / self.nb
        self.loss_P.backward()


    def backward_G(self):
        self.pool5_A = self.pool5_A_input + self.netG(self.A)
        self.pool5_A_input = Variable(self.pool5_A.data)
        pred_fake = self.netD.forward(self.pool5_A)

        self.loss_G = self.criterionAdv(pred_fake, True)
        self.loss_G.backward()

    def backward_D(self):
        pred_real = self.netD.forward(self.pool5_B_input)
        loss_D_real = self.criterionAdv(pred_real, True)

        pred_fake = self.netD.forward(self.pool5_A_input)
        loss_D_fake = self.criterionAdv(pred_fake, False)

        self.loss_D = (loss_D_real + loss_D_fake) * 0.5

        self.loss_D.backward()

    def backward_R(self):
        pool1 = self.deeplabPart1(self.A)
        self.predic_A_R = self.deeplabPart3(self.deeplabPart2(pool1) + self.netG(pool1))
        self.loss_R = self.criterionCE(self.predic_A_R, self.A_label)

        self.loss_R.backward()

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
        # Refine
        self.optimizer_R.zero_grad()
        self.backward_R()
        self.optimizer_R.step()

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, model_name, Iter, epoch, acc=[]):
        save_filename = '%s_model.pth' % (model_name)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save({
            'name':self.name(),
            'Iter': Iter,
            'epoch': epoch,
            'acc':acc,
            'state_dict_netG': self.netG.state_dict(),
            'state_dict_netD': self.netD.state_dict(),
            'state_dict_deeplabPart1': self.deeplabPart1.state_dict(),
            'state_dict_deeplabPart2':self.deeplabPart2.state_dict(),
            'state_dict_deeplabPart3': self.deeplabPart3.state_dict(),
            'optimizer_P':self.optimizer_P.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'optimizer_R': self.optimizer_R.state_dict(),
        }, save_path)

    def load(self, load_path):
        checkpoint = torch.load(load_path)
        self.netG.load_state_dict(checkpoint['state_dict_netG'])
        self.netD.load_state_dict(checkpoint['state_dict_netD'])
        self.deeplabPart1.load_state_dict(checkpoint['state_dict_deeplabPart1'])
        self.deeplabPart2.load_state_dict(checkpoint['state_dict_deeplabPart2'])
        self.deeplabPart3.load_state_dict(checkpoint['state_dict_deeplabPart3'])

        self.optimizer_P.load_state_dict(checkpoint['optimizer_P'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        self.optimizer_R.load_state_dict(checkpoint['optimizer_R'])
        for k,v in checkpoint['acc']:
            print('=================================================')
            print('accuracy: {1:.4f}\t'
                  'fg_accuracy: {2:.4f}\t'
                  'avg_precision: {3:.4f}\t'
                  'avg_recall: {4:.4f}\t'
                  'avg_f1score: {5:.4f}\t'
                  .format(v['accuracy'],v['fg_accuracy'],v['avg_precision'], v['avg_recall'], v['avg_f1score']))
            print('=================================================')

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        pass

    def train(self):
        self.deeplabPart1.train()
        self.deeplabPart2.train()
        self.deeplabPart3.train()
        self.netG.train()
        self.netD.train()

    def eval(self):
        self.deeplabPart1.eval()
        self.deeplabPart2.eval()
        self.deeplabPart3.eval()
        self.netG.eval()
        self.netD.eval()