import shutil
import torch
import time
import torch.nn as nn
from models.deeplab_gan_with_refine import deeplabGanWithRefine
from torch.autograd import Variable
from torch.utils import data
from loader.image_label_loader import imageLabelLoader
from loader.image_loader import imageLoader
from util.confusion_matrix import ConfusionMatrix
import util.makedirs as makedirs
import os
import torchvision.models as models
from util.log import Logger

def save_checkpoint(state, filename):
    torch.save(state, filename)


def update_confusion_matrix(matrix, output, target):
    values, indices = output.max(1)
    output = indices
    target = target.cpu().numpy()
    output = output.cpu().numpy()
    matrix.update(target, output)
    return matrix


def train(A_train_loader, B_train_loader, model, epoch):
    # switch to train mode
    model.train()
    for i, (A_image, A_label) in enumerate(A_train_loader):
        B_image = next(iter(B_train_loader))
        model.set_input({'A':A_image, 'A_label':A_label, 'B':B_image})
        model.optimize_parameters()
        output = model.output
        if i % args['print_freq'] == 0:
            matrix = ConfusionMatrix()
            update_confusion_matrix(matrix, output.data, A_label)

            logger.info('Epoch/Iter: [{epoch}/{Iter}]\t'
                  'loss: {loss:.4f}\t'
                  'acc: {accuracy:.4f}\t'
                  'fg_acc: {fg_accuracy:.4f}\t'
                  'avg_prec: {avg_precision:.4f}\t'
                  'avg_rec: {avg_recall:.4f}\t'
                  'avg_f1: {avg_f1core:.4f}\t'
                  'loss_G: {loss_G:.4f}\t'
                  'loss_D: {loss_D:.4f}\t'.format(
                epoch=epoch, Iter=i+epoch*len(A_train_loader), loss=model.loss_P.data[0], accuracy=matrix.accuracy(),
                fg_accuracy=matrix.fg_accuracy(), avg_precision=matrix.avg_precision(),
                avg_recall=matrix.avg_recall(), avg_f1core=matrix.avg_f1score(),
                loss_G=model.loss_G.data[0], loss_D=model.loss_D.data[0]))


def validate(val_loader, model, criterion, adaptation):
    # switch to evaluate mode
    run_time = time.time()
    matrix = ConfusionMatrix(args['label_nums'])
    loss = 0
    for i, (images, labels) in enumerate(val_loader):
        labels = labels.cuda(async=True)
        target_var = torch.autograd.Variable(labels, volatile=True)

        model.test(adaptation, images)
        output = model.output
        loss += criterion(output, target_var)/args['batch_size']
        matrix = update_confusion_matrix(matrix, output.data, labels)
    loss /= (i+1)
    run_time = time.time() - run_time
    logger.info('=================================================')
    logger.info('val:'
          'loss: {0:.4f}\t'
          'accuracy: {1:.4f}\t'
          'fg_accuracy: {2:.4f}\t'
          'avg_precision: {3:.4f}\t'
          'avg_recall: {4:.4f}\t'
          'avg_f1score: {5:.4f}\t'
          'run_time:{run_time:.2f}\t'
          .format(loss.data[0], matrix.accuracy(),
        matrix.fg_accuracy(), matrix.avg_precision(), matrix.avg_recall(), matrix.avg_f1score(),run_time=run_time))
    logger.info('=================================================')
    return matrix.all_acc()


def main():

    makedirs.mkdirs(os.path.join(args['checkpoints_dir'], args['name']))
    if len(args['device_ids']) > 0:
        torch.cuda.set_device(args['device_ids'][0])

    A_train_loader = data.DataLoader(imageLabelLoader(args['data_path'],dataName=args['domainA'], phase='train'), batch_size=args['batch_size'],
                                  num_workers=args['num_workers'], shuffle=True)
    A_val_loader = data.DataLoader(imageLabelLoader(args['data_path'], dataName=args['domainA'], phase='val'), batch_size=args['batch_size'],
                                num_workers=args['num_workers'], shuffle=False)

    B_train_loader = data.DataLoader(imageLoader(args['data_path'], dataName=args['domainB'], phase='train+unlabel'),
                                     batch_size=args['batch_size'],
                                     num_workers=args['num_workers'], shuffle=True)
    B_val_loader = data.DataLoader(imageLabelLoader(args['data_path'], dataName=args['domainB'], phase='val'),
                                   batch_size=args['batch_size'],
                                   num_workers=args['num_workers'], shuffle=False)
    model = deeplabGanWithRefine()
    model.initialize(args)

    # multi GPUS
    # model = torch.nn.DataParallel(model,device_ids=args['device_ids']).cuda()
    Iter = 0
    if args['resume']:
        if os.path.isfile(args['resume']):
            logger.info("=> loading checkpoint '{}'".format(args['resume']))
            model.load(args['resume'])
        else:
            print("=> no checkpoint found at '{}'".format(args['resume']))

    best_Ori_on_B = 0
    best_Ada_on_B = 0
    model.train()
    for epoch in range(args['n_epoch']):
        # train(A_train_loader, B_train_loader, model, epoch)
        # switch to train mode
        for i, (A_image, A_label) in enumerate(A_train_loader):
            Iter += 1
            B_image = next(iter(B_train_loader))
            model.set_input({'A': A_image, 'A_label': A_label, 'B': B_image})
            model.optimize_parameters()
            output = model.output
            if i % args['print_freq'] == 0:
                matrix = ConfusionMatrix()
                update_confusion_matrix(matrix, output.data, A_label)
                logger.info('Time: {time}\t'
                      'Epoch/Iter: [{epoch}/{Iter}]\t'
                      'loss: {loss:.4f}\t'
                      'loss_R: {loss_R:.4f}\t'
                      'acc: {accuracy:.4f}\t'
                      'fg_acc: {fg_accuracy:.4f}\t'
                      'avg_prec: {avg_precision:.4f}\t'
                      'avg_rec: {avg_recall:.4f}\t'
                      'avg_f1: {avg_f1core:.4f}\t'
                      'loss_G: {loss_G:.4f}\t'
                      'loss_D: {loss_D:.4f}\t'.format(
                    time=time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()),
                    epoch=epoch, Iter=Iter, loss=model.loss_P.data[0],
                    loss_R=model.loss_R.data[0], accuracy=matrix.accuracy(),
                    fg_accuracy=matrix.fg_accuracy(), avg_precision=matrix.avg_precision(),
                    avg_recall=matrix.avg_recall(), avg_f1core=matrix.avg_f1score(),
                    loss_G=model.loss_G.data[0], loss_D=model.loss_D.data[0]))

            if Iter % 1000 == 0:
                model.eval()
                acc_Ori_on_A = validate(A_val_loader, model, nn.CrossEntropyLoss(size_average=False), False)
                acc_Ori_on_B = validate(B_val_loader, model, nn.CrossEntropyLoss(size_average=False), False)
                acc_Ada_on_B = validate(B_val_loader, model, nn.CrossEntropyLoss(size_average=False), True)
                prec_Ori_on_B = acc_Ori_on_B['avg_f1score']
                prec_Ada_on_B = acc_Ada_on_B['avg_f1score']

                is_best = prec_Ori_on_B > best_Ori_on_B
                best_Ori_on_B = max(prec_Ori_on_B, best_Ori_on_B)
                if is_best:
                    model.save('best_Ori_on_B', Iter=Iter, epoch=epoch, acc={'acc_Ori_on_A':acc_Ori_on_A, 'acc_Ori_on_B':acc_Ori_on_B, 'acc_Ada_on_B':acc_Ada_on_B})

                is_best = prec_Ada_on_B > best_Ada_on_B
                best_Ada_on_B = max(prec_Ada_on_B, best_Ada_on_B)
                if is_best:
                    model.save('best_Ada_on_B', Iter=Iter, epoch=epoch, acc={'acc_Ori_on_A':acc_Ori_on_A, 'acc_Ori_on_B':acc_Ori_on_B, 'acc_Ada_on_B':acc_Ada_on_B})
                model.train()


if __name__ == '__main__':
    global args
    args = {
        'test_init':False,
        'label_nums':12,
        'l_rate':1e-8,
        'lr_gan': 0.00002,
        'lr_refine': 1e-6,
        'beta1': 0.5,
        'data_path':'datasets',
        'n_epoch':1000,
        'batch_size':10,
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
        'name': 'v3_5_t->s_Refine',
        'checkpoints_dir': 'checkpoints',
        'net_D': 'NoBNSinglePathdilationMultOutputNet',
        'use_lsgan': True,
        'resume':None#'checkpoints/v3_1/',
    }

    logger = Logger(
        log_file='./log/' + args['name'] + '-' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + '.log')
    logger.info('------------ Options -------------\n')
    for k, v in args.items():
        logger.info('%s: %s' % (str(k), str(v)))
    logger.info('-------------- End ----------------\n')
    main()