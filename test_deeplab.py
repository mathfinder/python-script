import torch
import scipy
import scipy.io as sio
import Image
from ourmodel.deeplab import Deeplab
import cv2
import numpy as np
# pretrained_dict = torch.load('/home/s-lab/rgh/AAAI/Gan/pytorch-caffe-darknet-convert/indoor(deeplab).pth')
# pretrained_dict = torch.load('/home/s-lab/rgh/AAAI/Gan/pytorch-caffe-darknet-convert/deeplab.pth')
pretrained_dict = torch.load('/home/s-lab/rgh/PyCharmProject/AAAI/pytorch/checkpoint/indoor_10.pth')
model = Deeplab()
model.init_weights(pretrained_dict=pretrained_dict)

# img = Image.open('/home/s-lab/rgh/AAAI/Gan/pytorch-caffe-darknet-convert/0007_test_6.jpg')
# img = np.array(img)
# img = img[:,:,::-1]
# img = img.transpose((2, 0, 1))
# img = img[np.newaxis,:,:,:]
# img = img.copy()
# img = torch.from_numpy(img)
img = Image.open('/home/s-lab/rgh/AAAI/Gan/pytorch-caffe-darknet-convert/0007_test_6.jpg')
img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
# RGB -> BGR
img = img.view(241, 121, 3)#.transpose(0,1).transpose(0,2).contiguous()
img = torch.stack([img[:,:,2], img[:,:,1], img[:,:,0]], 0)
img = img.view(1, 3, 241, 121)
img = img.float()

img = torch.autograd.Variable(img)
model.eval()
#out = model.test(img)
out = model(img)
# out batch c h w -> c h w
out = out.data.numpy().squeeze(axis=0)
out = out.argmax(axis=0)
sio.savemat('result/out.mat',{'result':out})
scipy.misc.imsave('result/test.png',out)




