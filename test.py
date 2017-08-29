import os
from torch.utils import data
from loader.image_label_loader import imageLabelLoader
from models.deeplab_gan_s2t_with_refine_4 import deeplabGanS2TWithRefine4
from util.confusion_matrix import ConfusionMatrix
import torch
import numpy as np
import scipy.misc
def color(label):
    bg = label == 0
    bg = bg.reshape(bg.shape[0], bg.shape[1])
    face = label == 1
    face = face.reshape(face.shape[0], face.shape[1])
    hair = label == 2
    hair = hair.reshape(hair.shape[0], hair.shape[1])
    Upcloth = label == 3
    Upcloth = Upcloth.reshape(Upcloth.shape[0], Upcloth.shape[1])
    Larm = label == 4
    Larm = Larm.reshape(Larm.shape[0], Larm.shape[1])
    Rarm = label == 5
    Rarm = Rarm.reshape(Rarm.shape[0], Rarm.shape[1])
    pants = label == 6
    pants = pants.reshape(pants.shape[0], pants.shape[1])
    Lleg = label == 7
    Lleg = Lleg.reshape(Lleg.shape[0], Lleg.shape[1])
    Rleg = label == 8
    Rleg = Rleg.reshape(Rleg.shape[0], Rleg.shape[1])
    dress = label == 9
    dress = dress.reshape(dress.shape[0], dress.shape[1])
    Lshoe = label == 10
    Lshoe = Lshoe.reshape(Lshoe.shape[0], Lshoe.shape[1])
    Rshoe = label == 11
    Rshoe = Rshoe.reshape(Rshoe.shape[0], Rshoe.shape[1])

    # bag = label == 12
    # bag = bag.reshape(bag.shape[0], bag.shape[1])

    # repeat 2nd axis to 3
    label = label.reshape(bg.shape[0], bg.shape[1], 1)
    label = label.repeat(3, 2)
    R = label[:, :, 2]
    G = label[:, :, 1]
    B = label[:, :, 0]
    R[bg] = 230
    G[bg] = 230
    B[bg] = 230

    R[face] = 255
    G[face] = 215
    B[face] = 0

    R[hair] = 80
    G[hair] = 49
    B[hair] = 49

    R[Upcloth] = 51
    G[Upcloth] = 0
    B[Upcloth] = 255

    R[Larm] = 2
    G[Larm] = 251
    B[Larm] = 49

    R[Rarm] = 141
    G[Rarm] = 255
    B[Rarm] = 212

    R[pants] = 160
    G[pants] = 0
    B[pants] = 255

    R[Lleg] = 0
    G[Lleg] = 204
    B[Lleg] = 255

    R[Rleg] = 191
    G[Rleg] = 255
    B[Rleg] = 248

    R[dress] = 255
    G[dress] = 182
    B[dress] = 185

    R[Lshoe] = 180
    G[Lshoe] = 122
    B[Lshoe] = 121

    R[Rshoe] = 202
    G[Rshoe] = 160
    B[Rshoe] = 57

    # R[bag] = 255
    # G[bag] = 1
    # B[bag] = 1
    return label
def update_confusion_matrix(matrix, output, target):
    values, indices = output.max(1)
    output = indices
    target = target.cpu().numpy()
    output = output.cpu().numpy()
    matrix.update(target, output)
    return matrix

def main():
    if len(args['device_ids']) > 0:
        torch.cuda.set_device(args['device_ids'][0])

    test_loader = data.DataLoader(imageLabelLoader(args['data_path'], dataName=args['domainB'], phase='test'),
                                   batch_size=args['batch_size'],
                                   num_workers=args['num_workers'], shuffle=False)
    gym = deeplabGanS2TWithRefine4()
    gym.initialize(args)
    gym.load('/home/ben/mathfinder/PROJECT/AAAI2017/our_Method/v3/deeplab_feature_adaptation/checkpoints/v3_s->t_Refine_4/best_Ori_on_B_model.pth')
    gym.eval()
    matrix = ConfusionMatrix(args['label_nums'])
    for i, (image, label) in enumerate(test_loader):
        label = label.cuda(async=True)
        target_var = torch.autograd.Variable(label, volatile=True)

        gym.test(False, image)
        output = gym.output

        """
        output = output.view(output.size()[1:]).data.cpu().numpy()
        output = output.argmax(0)
        output = output[:,:,np.newaxis]
        output = color(output)
        image = image.view(image.size()[1:]).cpu().numpy()
        image = image.transpose((1,2,0))

        label = label.view(label.size()[1:]).cpu().numpy()
        label = label[:, :, np.newaxis]
        label = color(label)


        output = np.concatenate((image, label, output),1)

        scipy.misc.imsave('/home/ben/mathfinder/PROJECT/AAAI2017/our_Method/v3/deeplab_feature_adaptation/datasets/bk/lip_redict/{}.png'.format(i),output)
        """
        matrix = update_confusion_matrix(matrix, output.data, label)
    print(matrix.all_acc())


if __name__ == "__main__":
    global args
    args = {
        'test_init':False,
        'label_nums':12,
        'l_rate':1e-8,
        'lr_gan': 0.00001,
        'lr_refine': 1e-6,
        'beta1': 0.5,
        'data_path':'datasets',
        'n_epoch':1000,
        'batch_size':1,
        'num_workers':10,
        'print_freq':10,
        'device_ids':[1],
        'domainA': 'Lip',
        'domainB': 'Indoor',
        'weigths_pool': 'pretrain_models',
        'pretrain_model': 'deeplab.pth',
        'fineSizeH':241,
        'fineSizeW':121,
        'input_nc':3,
        'name': 'v3_s->t_Refine_4',
        'checkpoints_dir': 'checkpoints',
        'net_D': 'NoBNSinglePathdilationMultOutputNet',
        'use_lsgan': True,
        'resume':None#'checkpoints/v3_1/',
    }
    main()