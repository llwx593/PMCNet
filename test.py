from models import *
import cv2
import numpy as np
from mindspore import Tensor, set_context, PYNATIVE_MODE
from mindspore import ops
from mindspore import load_checkpoint, load_param_into_net

num_classes = 2
image_size = (512,512)
save_dir='./result/'
base_dir = './data/polyp/test/'
dataset = 'polyp'
model_path = ""

image = cv2.imread(base_dir + '/images/' + '1.png', cv2.IMREAD_COLOR)
label = cv2.imread(base_dir + '/masks/' + '1.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, image_size, interpolation = cv2.INTER_NEAREST)
label = cv2.resize(label, image_size, interpolation = cv2.INTER_NEAREST)/255
image = np.asarray(image, np.float32)
image = image.transpose((2, 0, 1))
image = Tensor.from_numpy(image.astype(np.float32))
label = Tensor.from_numpy(label.astype(np.uint8))
image = ops.expand_dims(image, 0)

set_context(mode=PYNATIVE_MODE)
model = resnet34(num_classes)
if model_path != "":
    param_dict = load_checkpoint(model_path)
    load_param_into_net(model, param_dict)

model.set_train(False)
pred = model(image)

pred = pred[0,1,:,:]
pred = pred.asnumpy()
cv2.imwrite(save_dir+'Pre'+str(1)+'.jpg',pred[:,:]*255)
