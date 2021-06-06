import torch
import torchvision
from torch import optim
import torch.nn as nn
import numpy as np
import load_mnist
import os
from net import discriminator, generator
from torchvision import transforms

if __name__ == "__main__":

    if os.path.exists('cgan_images') is False:
        os.makedirs('cgan_images')

    criterion = nn.MSELoss()
    batch_size = 100
    z_dimension = 110
    D = discriminator()
    G = generator()
    train_set, labels = load_mnist.load_mnist("./dataset", kind='train')
    num_batchs = int(train_set.shape[0] / batch_size)
    d_optimizer = optim.Adam(D.parameters(), lr=0.0003)
    g_optimizer = optim.Adam(G.parameters(), lr=0.0003)

    count = 0
    epoch = 100
    gepoch = 1
    for i in range(epoch):
        for k in range(num_batchs):
            train_img = train_set[k * batch_size: (k + 1) * batch_size]
            real_labels = labels[k * batch_size: (k + 1) * batch_size]
            train_img = transforms.ToTensor()(train_img)
            train_img = transforms.Normalize(0.5, 0.5, 0.5)(train_img)
            labels_onehot = np.zeros((batch_size, 10))
            labels_onehot[np.arange(batch_size), real_labels] = 1
            labels_onehot = torch.from_numpy(labels_onehot)
            valid = torch.ones((batch_size, 1))
            invalid = torch.zeros((batch_size, 1))
            real_in = torch.cat((train_img.squeeze(), labels_onehot), dim=1)
            d_optimizer.zero_grad()

            real_out = D(real_in.float())
            real_loss = criterion(real_out, valid)

            z = torch.randn((batch_size, 100))
            fake_labels = np.random.randint(0, 10, batch_size)
            fake_onehot = np.zeros((batch_size, 10))
            fake_onehot[np.arange(batch_size), fake_labels] = 1
            fake_onehot = torch.from_numpy(fake_onehot).float()

            fake_img = G(z, fake_onehot)
            D_in = torch.cat((fake_img, fake_onehot), dim=1).float()
            fake_out = D(D_in.detach())
            fake_loss = criterion(fake_out, invalid)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            fake_img = G(z, fake_onehot)
            D_in = torch.cat((fake_img, fake_onehot), dim=1).float()
            fake_out = D(D_in)
            g_loss = criterion(fake_out, valid)
            g_loss.backward()
            g_optimizer.step()

            #
            # for j in range(gepoch):
            #     z = torch.randn((batch_size, 100))
            #     fake_labels = np.random.randint(0, 10, batch_size)
            #     fake_onehot = np.zeros((batch_size, 10))
            #     fake_onehot[np.arange(batch_size), fake_labels] = 1
            #     fake_onehot = torch.from_numpy(fake_onehot).float()
            #     g_optimizer.zero_grad()
            #
            #     fake_img = G(z, fake_onehot)
            #     D_in = torch.cat((fake_img, fake_onehot), dim=1).float()
            #     output = D(D_in)
            #     g_loss = criterion(output, valid)
            #     g_loss.backward()
            #     g_optimizer.step()
            print('Epoch [{}/{}], Batch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f}'.format(i, epoch, k, num_batchs, d_loss.data, g_loss.data))
        test_labels = torch.zeros((10, 10))
        for k in range(10):
            test_labels[k, k] = 1;
        z = torch.randn((10, 100))
        g_img = G(z, test_labels)
        g_img = g_img.view(-1, 1, 28, 28)
        torchvision.utils.save_image(g_img, './cgan_images/fake_images-{}.png'.format(i + 1), normalize=True)
