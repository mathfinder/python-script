import shutil
import torch
import time
import torch.nn as nn
from models.deeplab import Deeplab
from torch.autograd import Variable
from torch.utils import data
from loader.image_label_loader import imageLabelLoader
from util.confusion_matrix import ConfusionMatrix

import torchvision.models as models

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoint/model_best.pth.tar')


def update_confusion_matrix(matrix, output, target):
    values, indices = output.max(1)
    output = indices
    target = target.cpu().numpy()
    output = output.cpu().numpy()
    matrix.update(target, output)
    return matrix


def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        run_time = time.time()
        labels = labels.cuda(async=True)
        input_var = torch.autograd.Variable(images)
        target_var = torch.autograd.Variable(labels)
        # compute output
        output = model.forward(input_var)
        loss = criterion(output, target_var)/args['batch_size']
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % args['print_freq'] == 0:
            matrix = ConfusionMatrix()
            update_confusion_matrix(matrix, output.data, labels)
            run_time = time.time() - run_time
            print('Epoch/Iter: [{epoch}/{iter}]\t'
                  'loss: {loss:.4f}\t'
                  'acc: {accuracy:.4f}\t'
                  'fg_acc: {fg_accuracy:.4f}\t'
                  'avg_prec: {avg_precision:.4f}\t'
                  'avg_rec: {avg_recall:.4f}\t'
                  'avg_f1: {avg_f1:.4f}\t'
                  'run_time:{run_time:.2f}\t'.format(
                epoch=epoch, iter=i+epoch*len(train_loader), loss=loss.data[0], accuracy=matrix.accuracy(),
                fg_accuracy=matrix.fg_accuracy(), avg_precision=matrix.avg_precision(),
                avg_recall=matrix.avg_recall(), avg_f1core=matrix.avg_f1score(), run_time=run_time))


def validate(val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()
    run_time = time.time()
    matrix = ConfusionMatrix(args['label_nums'])
    loss = 0
    for i, (images, labels) in enumerate(val_loader):
        labels = labels.cuda(async=True)
        input_var = torch.autograd.Variable(images, volatile=True)
        target_var = torch.autograd.Variable(labels, volatile=True)
        # compute output
        output = model(input_var)
        loss += criterion(output, target_var)/args['batch_size']
        matrix = update_confusion_matrix(matrix, output.data, labels)
    loss /= (i+1)
    run_time = time.time() - run_time
    print('=================================================')
    print('val:'
          'loss: {0:.4f}\t'
          'accuracy: {1:.4f}\t'
          'fg_accuracy: {2:.4f}\t'
          'avg_precision: {3:.4f}\t'
          'avg_recall: {4:.4f}\t'
          'avg_f1score: {5:.4f}\t'
          'run_time:{run_time:.2f}\t'
          .format(loss.data[0], matrix.accuracy(),
        matrix.fg_accuracy(), matrix.avg_precision(), matrix.avg_recall(), matrix.avg_f1score(),run_time=run_time))
    print('=================================================')
    return matrix.avg_f1score()

def get_parameters(model, parameter_name):
    for name, param in model.named_parameters():
        if name in [parameter_name]:
            return param
def main():
    train_loader = data.DataLoader(imageLabelLoader(args['data_path'],dataName=args['domainB'], phase='train'), batch_size=args['batch_size'],
                                  num_workers=args['num_workers'], shuffle=True)
    val_loader = data.DataLoader(imageLabelLoader(args['data_path'], dataName=args['domainB'], phase='val'), batch_size=args['batch_size'],
                                num_workers=args['num_workers'], shuffle=False)
    model = Deeplab()
    print(model)
    if args['pretrain_model'] != '':
        pretrained_dict = torch.load(args['weigths_pool'] + '/' + args['pretrain_model'])
        model.weights_init(pretrained_dict=pretrained_dict)
    else:
        model.apply(weights_init())

    ignored_params = list(map(id, model.fc8_1.parameters()))
    ignored_params.extend(list(map(id, model.fc8_2.parameters())))
    ignored_params.extend(list(map(id, model.fc8_3.parameters())))
    ignored_params.extend(list(map(id, model.fc8_4.parameters())))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())

    optimizer = torch.optim.SGD([
        {'params': base_params},
        {'params': get_parameters(model.fc8_1, 'weight'), 'lr': args['l_rate'] * 10},
        {'params': get_parameters(model.fc8_2, 'weight'), 'lr': args['l_rate'] * 10},
        {'params': get_parameters(model.fc8_3, 'weight'), 'lr': args['l_rate'] * 10},
        {'params': get_parameters(model.fc8_4, 'weight'), 'lr': args['l_rate'] * 10},
        {'params': get_parameters(model.fc8_1, 'bias'), 'lr': args['l_rate'] * 20},
        {'params': get_parameters(model.fc8_2, 'bias'), 'lr': args['l_rate'] * 20},
        {'params': get_parameters(model.fc8_3, 'bias'), 'lr': args['l_rate'] * 20},
        {'params': get_parameters(model.fc8_4, 'bias'), 'lr': args['l_rate'] * 20},
    ], lr=args['l_rate'], momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(size_average=False).cuda()
    # multi GPUS
    model = torch.nn.DataParallel(model,device_ids=args['device_ids']).cuda()
    best_prec = 0
    for epoch in range(args['n_epoch']):
        train(train_loader, model, criterion, optimizer, epoch)

        if epoch > 0 and epoch % 9 == 0:
            prec = validate(val_loader, model, criterion)
            is_best = prec > best_prec
            best_prec = max(prec, best_prec)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'deeplab(indoor)',
                'state_dict': model.state_dict(),
                'best_prec1': best_prec,
                'optimizer': optimizer.state_dict(),
            }, is_best,filename='./checkpoint/indoor_epoch_'+str(epoch)+'.pth.tar')
        #break
if __name__ == '__main__':
    global args
    args = {
        'test_init': False,
        'label_nums': 12,
        'l_rate': 1e-8,
        'data_path': 'datasets',
        'n_epoch': 1000,
        'batch_size': 10,
        'num_workers': 10,
        'print_freq': 10,
        'device_ids': [0],
        'domainA': 'Lip',
        'domainB': 'Indoor',
        'weigths_pool': 'pretrain_models',
        'pretrain_model': 'deeplab.pth',
        'loadSizeH': 241,
        'loadSizeW': 121,
        'fineSizeH': 241,
        'fineSizeW': 121,
        'name':'v3',
        'checkpoints_dir':'checkpoints',
    }
    main()