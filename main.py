import os
import torch
import torch.nn as nn
from dataloader import MelodyDataset
from torch.utils.data import DataLoader
from signal_process import signal_process
import torch.nn.functional as F
import numpy as np
import random
from time import time
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')


# TODO : IMPORTANT !!! Please change it to True when you submit your code
is_test_mode = True

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO : IMPORTANT !!! Please specify the path where your best model is saved
# example : ckpt/model.pth
ckpt_dir = 'ckpt'
best_saved_model = 'top_accuracy.pth'
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
restore_path = os.path.join(ckpt_dir, best_saved_model)

# Data paths
# TODO : IMPORTANT !!! Do not change metadata_path. Test will be performed by replacing this file.
metadata_path = 'metadata.csv'
audio_dir = 'audio'

# TODO : Declare additional hyperparameters
# not fixed (change or add hyperparameter as you like)
# you may train with num_label = 40 but print accuracy with 13 labels in test mode [ use dataloader mapping dict ]
n_epochs = 100
batch_size = 24
num_label = 13
method = 'logmelspectrogram'
sr = 11025
alpha = 0.5

# continue_path = './ckpt/top_accuracy.pth'

# reproduce result
seed = 929
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, leaky_relu_slope=0.01):
        super(ResBlock, self).__init__()

        # BN / LReLU / MaxPool layer before the conv layer - see Figure 1b in the paper
        self.pre_conv = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(leaky_relu_slope),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),  # apply downsampling on the y axis only
            nn.Dropout(0.5)
        )

        # Conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )

        # 1 x 1 convolution layer to match the feature map dimensions
        self.conv1by1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pre_conv(x)
        x_conv = self.conv(x)
        out = x_conv + self.conv1by1(x)

        return out


# TODO : Build your model here
class YourModel(nn.Module):
    def __init__(self, c, leaky_relu_slope=0.01):
        super(YourModel, self).__init__()

        ### MAIN NETWORK ###

        # ConvBlock
        self.ConvBlock = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(c, c, 3, padding=1)
        )

        # ResBlocks
        self.ResBlock1 = ResBlock(in_channels=c, out_channels=2 * c)
        self.ResBlock2 = ResBlock(in_channels=2 * c, out_channels=4 * c)
        self.ResBlock3 = ResBlock(in_channels=4 * c, out_channels=8 * c)
        self.ResBlock4 = ResBlock(in_channels=8 * c, out_channels=16 * c)

        # PoolBlock
        self.PoolBlock = nn.Sequential(
            nn.BatchNorm2d(16 * c),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(0.5)
        )

        self.blstm = nn.LSTM(input_size=16 * c, hidden_size=4 * c,
                             num_layers=3, bidirectional=True,
                             batch_first=True)

        self.classifier = nn.Linear(in_features=8 * c, out_features=num_label)

        ### AUXILIARY NETWORK ###

        # maxpool layers (for auxiliary network inputs)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 64), stride=(1, 64))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 16), stride=(1, 16))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))

        self.blstm2 = nn.LSTM(input_size=44 * c, hidden_size=4 * c,
                              num_layers=3, bidirectional=True,
                              batch_first=True)

        # binary classifier
        self.detector = nn.Linear(in_features=8 * c, out_features=2)

        # initialize weights
        self.apply(self.init_weights)

    def forward(self, x):

        ### PITCH CLASSFIER ###

        x = x.transpose(1, 2).unsqueeze(1)

        conv_out = self.ConvBlock(x)
        res1_out = self.ResBlock1(conv_out)
        res2_out = self.ResBlock2(res1_out)
        res3_out = self.ResBlock3(res2_out)
        res4_out = self.ResBlock4(res3_out)
        pool_out = self.PoolBlock(res4_out)

        x = pool_out.permute(0, 2, 1, 3).contiguous().view((batch_size, 240, -1))
        x, _ = self.blstm(x)
        o_m = self.classifier(x)

        ### VOICE DETECTOR ###

        mp1_out = self.maxpool1(res2_out)
        mp2_out = self.maxpool2(res3_out)
        mp3_out = self.maxpool3(res4_out)

        y = torch.cat((mp1_out, mp2_out, mp3_out, pool_out), dim=1)
        y = y.permute(0, 2, 1, 3).contiguous().view((batch_size, 240, -1))
        y, _ = self.blstm2(y)
        o_v = self.detector(y)

        # extract voice/non-voice from softmax output of main network
        v, nv = torch.split(o_m, [num_label-1, 1], dim=2)
        o_mv = torch.cat((torch.sum(v, dim=2, keepdim=True), nv), dim=2)
        o_sv = o_v + o_mv

        return {'o_m': o_m, 'o_sv': o_sv}

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.LSTMCell):
            for p in m.parameters():
                if p.data is None:
                    continue

                if len(p.shape) >= 2:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.normal_(p.data)


# Save model with state_dict
def save_checkpoint(epoch, model, optimizer, path):
    state = {
        'epoch': epoch,
        'net_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict()
    }
    torch.save(state, path)


if not is_test_mode:

    # Write the result to tensorboard
    writer = SummaryWriter('./tensorboard')

    # Load Dataset and Dataloader
    train_dataset = MelodyDataset(metadata_path=metadata_path, audio_dir=audio_dir, sr=sr, split='training', num_label=num_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16)
    valid_dataset = MelodyDataset(metadata_path=metadata_path, audio_dir=audio_dir, sr=sr, split='validation', num_label=num_label)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=16)

    model = YourModel(c=16, leaky_relu_slope=0.01)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # focal loss
    focal_loss = torch.hub.load(
        'adeelh/pytorch-multi-class-focal-loss',
        model='FocalLoss',
        gamma=4,
        reduction='mean',
        force_reload=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    # Learning rate scheduler: ReduceLROnPlateau
    scheduler_list = [
        optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-10),
        optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1)),
        optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.5),
        optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[2, 5, 10, 11, 28], gamma=0.5),
        optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96),
        optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=0)
    ]
    scheduler = scheduler_list[0]
    scheduler_name = scheduler.__class__.__name__

    epoch_continue = 0
    # # continue training with saved model
    # checkpoint = torch.load(continue_path)
    # model.load_state_dict(checkpoint['net_state_dict'])
    # optimizer.load_state_dict(checkpoint['optim_state_dict'])
    # epoch_continue = checkpoint['epoch'] + 1

    # to compare the result of current epoch with previous best accuracy
    top_accuracy = 0

    # Training and Validation
    for epoch in range(n_epochs):
        epoch += epoch_continue
        epoch_start = time()
        model.train()

        train_correct = 0
        train_loss = 0

        # print learning rate of current epoch
        lr = optimizer.param_groups[0]['lr']
        print('==== Epoch:', epoch, ', LR:', lr)

        for idx, (features, labels, is_voice) in enumerate(train_loader):
            optimizer.zero_grad()
            features = signal_process(features, sr=sr, method=method).to(device)
            labels = labels.to(device)
            is_voice = is_voice.to(device)

            output = model(features)
            # o_m is pitch classification output of main network
            # o_sv is voice detection output
            l_pitch = criterion(output['o_m'].view(-1, num_label), labels.view(-1))
            l_voice = criterion(output['o_sv'].view(-1, 2), is_voice.view(-1))
            # weighted loss
            loss = l_pitch + alpha * l_voice
            train_loss = train_loss + loss.item()
            loss.backward()
            optimizer.step()

            preds = output['o_m'].argmax(dim=-1, keepdim=True)
            train_correct += (preds.squeeze() == labels).float().sum()

        print("==== Epoch: %d, Train Loss: %.2f, Train Accuracy: %.3f" % (
            epoch, train_loss / len(train_loader), train_correct / (240*len(train_loader)*24)))

        model.eval()

        valid_correct = 0
        valid_loss = 0

        for idx, (features, labels, is_voice) in enumerate(valid_loader):
            features = signal_process(features, sr=sr, method=method)
            labels = labels.to(device)
            is_voice = is_voice.to(device)

            output = model(features)
            l_pitch = criterion(output['o_m'].view(-1, num_label), labels.view(-1))
            l_voice = criterion(output['o_sv'].view(-1, 2), is_voice.view(-1))
            loss = l_pitch + alpha * l_voice
            valid_loss = valid_loss + loss.item()

            preds = output['o_m'].argmax(dim=-1, keepdim=True)
            valid_correct += (preds.squeeze() == labels).float().sum()

        print("==== Epoch: %d, Valid Loss: %.2f, Valid Accuracy: %.3f" % (
            epoch, valid_loss / len(valid_loader), valid_correct / (240*len(valid_loader)*24)))

        # Write to tensorboard
        writer.add_scalars('loss/training+validation', {"loss_training": train_loss / len(train_loader),
                                                        "loss_validation": valid_loss / len(valid_loader)}, epoch)
        writer.add_scalars('accuracy/training+validation', {"accuracy_training": train_correct / (240*len(train_loader)*24),
                                                            "accuracy_validation": valid_correct / (240*len(valid_loader)*24)}, epoch)
        writer.add_scalar('lr/{}'.format(scheduler_name), lr, epoch)

        # learning rate scheduler
        scheduler.step(metrics=valid_loss)

        train_accuracy = train_correct / (240*len(train_loader)*24)
        valid_accuracy = valid_correct / (240*len(valid_loader)*24)

        # Save the model of current epoch with train_accuracy and valid_accuracy
        torch.save(model, "./ckpt/%d_%.4f_%.4f.pth" % (epoch, train_accuracy, valid_accuracy))
        save_checkpoint(epoch, model, optimizer, "./ckpt/%d_%.4f_%.4f_continue.pth" % (epoch, train_accuracy, valid_accuracy))

        # Save the model which results the best accuracy
        if valid_accuracy > top_accuracy:
            torch.save(model, './ckpt/top_accuracy.pth')
            top_accuracy = valid_accuracy

        print('TIME: %6.3f' % (time() - epoch_start))

    writer.close()


elif is_test_mode:

    # Load Dataset and Dataloader
    test_dataset = MelodyDataset(metadata_path=metadata_path, audio_dir=audio_dir, sr=sr, split='test', num_label=num_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Restore model
    model = torch.load(restore_path).to(device)
    print('==== Model restored : %s' % restore_path)

    # TODO: IMPORTANT!! MUST CALCULATE ACCURACY ! You may change this part, but must print accuracy in the right manner
    # Calculate accuracy by 13 labels - use dataloader mapping dict

    test_correct = 0

    for features, labels, is_voice in test_loader:
        features = signal_process(features, sr=sr, method=method).to(device)
        labels = labels.to(device)

        output = model(features)
        preds = output['o_m'].argmax(dim=-1, keepdim=True)
        test_correct += (preds.squeeze() == labels).float().sum()

    print("=== Test accuracy: %.3f" % (test_correct / (240*len(test_dataset))))