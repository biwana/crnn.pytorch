import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import numpy as np
import csv
import os
import models.crnn as crnn


model_path = './data/crnn.pth'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV'
labels = np.genfromtxt("title30cat-east-cut-test.txt", delimiter=" ", dtype=str)[:5]

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
for l in labels:
    print(l)
    image = Image.open(l).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    imavar = Variable(image)

    model.eval()
    preds = model(imavar)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))
    image.close()
    np.savetxt("output/%s.txt"%l, sim_pred)
