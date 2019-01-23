import random

import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

#ARCHITECTURE FOR N = 16384 SIGNALS
class DCGAN_Audio_Straight(nn.Module):
    def __init__(self, nz=32, ngf=64, output_size=16384, nc=1, num_measurements=1000, cuda = True):
        super(DCGAN_Audio_Straight, self).__init__()
        self.nc = nc
        self.output_size = output_size
        self.CUDA = cuda

        # Deconv Layers: (in_channels, out_channels, kernel_size, stride, padding, bias = false)
        # Inputs: R^(N x Cin x Lin), Outputs: R^(N, Cout, Lout) s.t. Lout = (Lin - 1)*stride - 2*padding + kernel_size

        self.conv1 = nn.ConvTranspose1d(nz, ngf, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm1d(ngf)
        # LAYER 1: input: (random) zϵR^(nzx1), output: x1ϵR^(64x4) (channels x length)

        self.conv2 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm1d(ngf)
        # LAYER 2: input: x1ϵR^(64x4), output: x2ϵR^(64x8) (channels x length)

        self.conv3 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn3 = nn.BatchNorm1d(ngf)
        # LAYER 3: input: x1ϵR^(64x8), output: x2ϵR^(64x16) (channels x length)

        self.conv4 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn4 = nn.BatchNorm1d(ngf)
        # LAYER 4: input: x1ϵR^(64x16), output: x2ϵR^(64x32) (channels x length)

        self.conv5 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn5 = nn.BatchNorm1d(ngf)
        # LAYER 5: input: x2ϵR^(64x32), output: x3ϵR^(64x64) (channels x length)

        self.conv6 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn6 = nn.BatchNorm1d(ngf)
        # LAYER 6: input: x3ϵR^(64x64), output: x4ϵR^(64x128) (channels x length)

        self.conv7 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn7 = nn.BatchNorm1d(ngf)
        # LAYER 7: input: x4ϵR^(64x128), output: x5ϵR^(64x256) (channels x length)

        self.conv8 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn8 = nn.BatchNorm1d(ngf)
        # LAYER 8: input: x5ϵR^(64x256), output: x6ϵR^(64x512) (channels x length)

        self.conv9 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn9 = nn.BatchNorm1d(ngf)
        # LAYER 9: input: x5ϵR^(64x512), output: x6ϵR^(64x1024) (channels x length)

        self.conv10 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn10 = nn.BatchNorm1d(ngf)
        # LAYER 10: input: x5ϵR^(64x1024), output: x6ϵR^(64x2048) (channels x length)

        self.conv11 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn11 = nn.BatchNorm1d(ngf)
        # LAYER 11: input: x5ϵR^(64x2048), output: x6ϵR^(64x4096) (channels x length)

        self.conv12 = nn.ConvTranspose1d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn12 = nn.BatchNorm1d(ngf)
        # LAYER 12: input: x5ϵR^(64x4096), output: x6ϵR^(64x8192) (channels x length)

        self.conv13 = nn.ConvTranspose1d(ngf, nc, 4, 2, 1, bias=False)  # output is image
        # LAYER 13: input: x6ϵR^(64x8192), output: (sinusoid) G(z,w)ϵR^(ncx16384) (channels x length)

        self.fc = nn.Linear(output_size * nc, num_measurements, bias=False)  # output is A; measurement matrix
        # each entry should be drawn from a Gaussian (random noisy measurements)
        # don't compute gradient of self.fc! memory issues

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = torch.tanh(self.conv13(x))

        return x

    def measurements(self, x):
        # this gives the image - make it a single row vector of appropriate length
        y = self.forward(x).view(1, -1)

        meas = self.fc(y).view(-1, 1)

        if self.CUDA:
            return meas.cuda()
        else:
            return meas

#ARCHITECTURE FOR N = 1024 SIGNALS
class DCGAN_Short(nn.Module):
    def __init__(self, nz=32, ngf=128, output_size=1024, nc=1, num_measurements=100, cuda = True):
        super(DCGAN_Short, self).__init__()
        self.nc = nc
        self.output_size = output_size
        self.CUDA = cuda

        # Deconv Layers: (in_channels, out_channels, kernel_size, stride, padding, bias = false)
        # Inputs: R^(N x Cin x Lin), Outputs: R^(N, Cout, Lout) s.t. Lout = (Lin - 1)*stride - 2*padding + kernel_size

        self.conv1 = nn.ConvTranspose1d(nz, ngf, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm1d(ngf)
        # LAYER 1: input: (random) zϵR^(nzx1), output: x1ϵR^(128x4) (channels x length)

        self.conv2 = nn.ConvTranspose1d(ngf, ngf, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(ngf)
        # LAYER 2: input: x1ϵR^(128x4), output: x2ϵR^(128x8) (channels x length)

        self.conv3 = nn.ConvTranspose1d(ngf, ngf, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(ngf)
        # LAYER 3: input: x1ϵR^(128x8), output: x2ϵR^(128x16) (channels x length)

        self.conv4 = nn.ConvTranspose1d(ngf, ngf, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm1d(ngf)
        # LAYER 4: input: x1ϵR^(128x16), output: x2ϵR^(128x32) (channels x length)

        self.conv5 = nn.ConvTranspose1d(ngf, ngf, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm1d(ngf)
        # LAYER 5: input: x2ϵR^(128x32), output: x3ϵR^(128x64) (channels x length)

        self.conv6 = nn.ConvTranspose1d(ngf, ngf, 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm1d(ngf)
        # LAYER 6: input: x3ϵR^(128x64), output: x4ϵR^(128x128) (channels x length)

        self.conv7 = nn.ConvTranspose1d(ngf, ngf, 4, 2, 1, bias=False)
        self.bn7 = nn.BatchNorm1d(ngf)
        # LAYER 7: input: x4ϵR^(128x128), output: x5ϵR^(128x256) (channels x length)

        self.conv8 = nn.ConvTranspose1d(ngf, ngf, 4, 2, 1, bias=False)
        self.bn8 = nn.BatchNorm1d(ngf)
        # LAYER 8: input: x5ϵR^(128x256), output: x6ϵR^(128x512) (channels x length)

        self.conv9 = nn.ConvTranspose1d(ngf, nc, 4, 2, 1, bias=False)  # output is image
        # LAYER 9: input: x6ϵR^(128x512), output: (sinusoid) G(z,w)ϵR^(ncx1024) (channels x length)

        self.fc = nn.Linear(output_size * nc, num_measurements, bias=False)  # output is A; measurement matrix
        # each entry should be drawn from a Gaussian (random noisy measurements)
        # don't compute gradient of self.fc! memory issues

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = torch.tanh(self.conv9(x))

        return x

    def measurements(self, x):
        # this gives the image - make it a single row vector of appropriate length
        y = self.forward(x).view(1, -1)

        meas = self.fc(y).view(-1, 1)

        if self.CUDA:
            return meas.cuda()
        else:
            return meas

#Given measurement matrix A and observed measurements y, return estimate of x
def run_DIP(A, y, dtype, NGF = 64, nz = 32, LR = 5e-4, MOM = 0.9, WD = 1e-4, num_channels = 1, output_size = 16384, num_measurements = 1000, CUDA = True, num_iter = 3000, alpha_tv = 1e-4, get_mse=False, true_signal = []):

    y = torch.Tensor(y)  #convert the input measurements to CUDA
    y = Variable(y.type(dtype))

    net = DCGAN_Audio_Straight(ngf = NGF, output_size = output_size, num_measurements = num_measurements, cuda=CUDA, nc=num_channels)
    net.fc.requires_grad = False
    net.fc.weight.data = torch.Tensor(A)

    allparams = [x for x in net.parameters()]  # specifies which to compute gradients of
    allparams = allparams[:-1]  # get rid of last item in list (fc layer) because it's memory intensive

    z = Variable(torch.zeros(nz).type(dtype).view(1, nz, 1))  # Define input seed z as Torch variable, normalize
    z.data.normal_().type(dtype)
    z.requires_grad = False

    if CUDA:
        net.cuda()

    optim = torch.optim.RMSprop(allparams, lr=LR, momentum=MOM, weight_decay=WD)

    if CUDA:  # move measurements to GPU if possible
        y = y.cuda()

    x_hat = np.zeros((output_size, num_channels))

    mse_log = []

    for i in range(num_iter):
        optim.zero_grad()  # clears gradients of all optimized variables
        out = net(z)  # produces wave (in form of data tensor) i.e. G(z,w)

        #loss = mse(net.measurements(z), y)  # calculate loss between AG(z,w) and Ay
        loss = MSE_TV_LOSS(net.measurements(z), y, alpha_tv, dtype)

        wave = out[0].detach().reshape(-1, num_channels).cpu()

        if get_mse:
            mse_log.append(np.mean((np.squeeze(true_signal) - np.squeeze(wave.numpy())) ** 2))

        if (i == num_iter - 1):
            x_hat = wave

        # if (i >= num_iter/2):  # if optimzn has converged, exit descent
        #     should_exit = exit_condition(mse_log[-exit_window:])
        #     if should_exit == True:
        #         x_hat = wave
        #         print("Early Stop: ", i)
        #         break

        loss.backward()
        optim.step()

    if get_mse:
        return [x_hat.numpy(), mse_log]

    return x_hat.numpy()

#Given measurement matrix A and observed measurements y, return estimate of x
def run_DIP_short(A, y, dtype, NGF = 64, nz = 32, LR = 1e-4, MOM = 0.9, WD = 1e-1, num_channels = 1, output_size = 1024, num_measurements = 100, CUDA = True, num_iter = 3000, alpha_tv = 1e-1, get_mse=False, true_signal=[]):

    y = torch.Tensor(y)  #convert the input measurements to CUDA
    y = Variable(y.type(dtype))

    net = DCGAN_Short(ngf = NGF, output_size = output_size, num_measurements = num_measurements, cuda=CUDA, nc=num_channels)
    net.fc.requires_grad = False
    net.fc.weight.data = torch.Tensor(A)

    allparams = [x for x in net.parameters()]  # specifies which to compute gradients of
    allparams = allparams[:-1]  # get rid of last item in list (fc layer) because it's memory intensive

    z = Variable(torch.zeros(nz).type(dtype).view(1, nz, 1))  # Define input seed z as Torch variable, normalize
    z.data.normal_().type(dtype)
    z.requires_grad = False

    if CUDA:
        net.cuda()

    optim = torch.optim.RMSprop(allparams, lr=LR, momentum=MOM, weight_decay=WD)

    if CUDA:  # move measurements to GPU if possible
        y = y.cuda()

    x_hat = np.zeros((output_size, num_channels))

    mse_log = []

    for i in range(num_iter):
        optim.zero_grad()  # clears graidents of all optimized variables
        out = net(z)  # produces wave (in form of data tensor) i.e. G(z,w)

        #loss = mse(net.measurements(z), y)  # calculate loss between AG(z,w) and Ay
        loss = MSE_TV_LOSS(net.measurements(z), y, alpha_tv, dtype)

        wave = out[0].detach().reshape(-1, num_channels).cpu()

        if get_mse:
            mse_log.append(np.mean((np.squeeze(true_signal) - np.squeeze(wave.numpy()))**2))

        if (i == num_iter - 1):
            x_hat = wave

        # if (i >= num_iter/2):  # if optimzn has converged, exit descent
        #     should_exit = exit_condition(mse_log[-exit_window:])
        #     if should_exit == True:
        #         x_hat = wave
        #         print("Early Stop: ", i)
        #         break

        loss.backward()
        optim.step()

    if get_mse:
        return [x_hat.numpy(), mse_log]

    return x_hat.numpy()

#TV Loss for Network training
def MSE_TV_LOSS (x_hat, x, alpha, dtype):

    x_hat_shift = x_hat.detach().cpu().numpy()
    x_hat_shift = np.roll(x_hat_shift, 1) #shift x_hat right by 1 to do TV
    x_hat_shift = torch.Tensor(x_hat_shift)
    x_hat_shift = Variable(x_hat_shift.type(dtype)) #convert back to torch tensor to use gradient

    tv = x_hat - x_hat_shift
    tv[0,0] = 0
    tv = abs(tv)

    mse = torch.nn.MSELoss(reduction='sum').type(dtype)
    mseloss = mse(x_hat, x)

    return mseloss + alpha*torch.sum(tv)
