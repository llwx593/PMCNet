import numpy as np

from models import *
from dataset import *
from utils import *

import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.dataset import GeneratorDataset
from mindspore.train.callback import LossMonitor
from mindspore.train.model import Model

batch_size = 32
num_classes = 2
image_size = (256, 256)
base_dir = './data/drive/'
dataset = 'drive'
train_num = 500  # 125
max_epoch = 200

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)
context.set_context(device="0")

train_data = MyDataSet(base_dir+'train/','train.txt',train_num, image_size, dataset)
train_ds = GeneratorDataset(train_data)
train_ds = train_ds.batch(batch_size)

test_data = MyDataSet(base_dir+'test/','test.txt',900, image_size, dataset)
test_ds = GeneratorDataset(test_data)
test_ds = test_ds.batch(batch_size)

net = resnet34(num_classes)

loss_func = SoftmaxCrossEntropyWithLogits()

lr = Tensor(get_lr(global_step=0, total_epochs=max_epoch, steps_per_epoch=train_ds.get_dataset_size()))

optimizer = nn.Adam(net.trainable.parameters(), lr, moentum=(0.9,0.99), weight_decay=0.0001)

model = Model(net, loss_fn=loss_func, optimizer=optimizer, metrics={'acc'})
loss_cb = LossMonitor()
cb = [loss_cb]

model.train(max_epoch, train_ds, callbacks=cb)

output = model.eval(test_ds)
print(f'Evaluation result: {output}.')            
            