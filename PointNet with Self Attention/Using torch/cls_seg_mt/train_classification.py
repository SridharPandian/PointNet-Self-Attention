from __future__ import print_function
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from datasets import PartDataset
from pointnet import PointNetCls
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
from torch.optim.lr_scheduler import LambdaLR


def plot_graphs(graph_dict, save_dict=True):
    # train and test loss curves
    plt.plot(graph_dict['epochs'], graph_dict['train_loss'], label='train_loss')
    plt.plot(graph_dict['epochs'], graph_dict['val_loss'], label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plots_dir, 'loss.png'))
    plt.close()

    # train and test accuracies
    plt.plot(graph_dict['epochs'], graph_dict['train_acc'], label='train_acc')
    plt.plot(graph_dict['epochs'], graph_dict['val_acc'], label='val_acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plots_dir, 'accuracy.png'))
    plt.close()

    if save_dict:
        j = json.dumps(graph_dict)
        f = open(os.path.join(plots_dir, "stats.json"), "w")
        f.write(j)
        f.close()

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='data/train', help='path to input data')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--num_points', type=int, default=1000, help='number of points to resample')
parser.add_argument('--min_points', type=int, default=0, help='smallest point cloud')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='models/cls',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--lr_decay_rate', type=float, default = 0.05,  help='model path')

if __name__ == '__main__': 
    opt = parser.parse_args()
    print (opt)
    model_dir = opt.outf

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using device: {}'.format(device))

    blue = lambda x:'\033[94m' + x + '\033[0m'

    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    dataset = PartDataset(root = os.path.join(opt.input_path, 'train'),
                        task='classification',
                        npoints = opt.num_points,
                        min_pts=0,
                        load_in_memory=True,
                        num_seg_class=5)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=opt.batchSize,
                                            shuffle=True,
                                            num_workers=int(opt.workers))

    val_dataset = PartDataset(root = os.path.join(opt.input_path, 'val'),
                            task='classification',
                            mode = 'val',
                            npoints = opt.num_points,
                            min_pts = 0,
                            load_in_memory=True,
                            num_seg_class=5)
    valdataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=opt.batchSize,
                                                shuffle=True,
                                                num_workers=int(opt.workers))

    print('train: {} val: {}'.format(len(dataset), len(val_dataset)))
    num_classes = len(dataset.classes)
    print('classes', num_classes)


    classifier = PointNetCls(k = num_classes).to(device)
    start_epoch=-1

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))
        # TODO update start_epoch from pre-trained

    # for param_tensor in classifier.state_dict():
    #         print(param_tensor, "\t", classifier.state_dict()[param_tensor].size())

    optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, classifier.parameters()),
                        lr=0.01,
                        momentum=0.9)
    lambda_lr = lambda epoch: 1 / (1 + (opt.lr_decay_rate * epoch))
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr, last_epoch=start_epoch)

    num_batch = len(dataset)/opt.batchSize
    num_val_batch = len(val_dataset)/opt.batchSize
    n_log = 100

    epochs = []
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []


    plots_dir = os.path.join(model_dir, 'plots')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    for epoch in range(opt.nepoch):
        epoch_train_loss = 0
        epoch_test_loss = 0
        test_batch_num = 0

        lr_scheduler.step()

        total_train_correct = 0
        total_train_loss = 0
        classifier.train()
        for i, data in enumerate(dataloader, 0):
            points, target = data
            points, target = points.to(device, non_blocking=True), target[:, 0].to(device, non_blocking=True)
            points = points.transpose(2,1)
            optimizer.zero_grad()
            pred, _ = classifier(points)
            loss = F.nll_loss(pred, target.long())
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            total_train_correct += correct.item()
            total_train_loss += loss.item()

            if i%n_log == 0:
                print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize)))

            del loss

        epoch_train_acc = float(total_train_correct) / float(len(dataset))
        epoch_train_loss = float(total_train_loss) / float(num_batch)
        print('train_accuracy: {}'.format(epoch_train_acc))
        print('train_loss: {}'.format(epoch_train_loss))

        total_test_correct = 0
        total_test_loss = 0
        classifier.eval()
        for j, data in enumerate(valdataloader, 0):
            test_batch_num += 1
            points, target = data
            points, target = points.to(device, non_blocking=True), target[:, 0].to(device, non_blocking=True)
            points = points.transpose(2, 1)
            pred, _ = classifier(points)
            loss = F.nll_loss(pred, target.long())
            epoch_test_loss += loss.item()
            pred_choice = pred.data.max(1)[1]
            total_test_correct += pred_choice.eq(target.long().data).cpu().sum().item()
            total_test_loss += loss.item()
        epoch_test_acc = float(total_test_correct) / float(len(val_dataset))
        epoch_test_loss = float(total_test_loss) / float(num_val_batch)
        print('val accuracy: {}'.format(epoch_test_acc))
        print('val_loss: {}'.format(epoch_test_loss))

        # for param_tensor in classifier.state_dict():
        #     print(param_tensor, "\t", classifier.state_dict()[param_tensor].size())

        epochs.append(epoch)
        train_acc.append(epoch_train_acc)
        train_loss.append(epoch_train_loss)
        test_acc.append(epoch_test_acc)
        test_loss.append(epoch_test_loss)
        plot_graphs({'epochs':epochs, 'train_acc':train_acc, 'train_loss':train_loss, 'val_acc':test_acc, 'val_loss':test_loss})
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (model_dir, epoch))
