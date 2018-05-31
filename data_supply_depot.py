import numpy as np
import scipy.misc as misc
import cv2


def supply(batch_num, batch_size ,data_files,size,mean):


    x_batch = np.zeros((batch_size, size, size, 3))
    x_batch_2 = np.zeros((batch_size, size, size, 3))
    y_batch = np.zeros((batch_size, size, size))
    z_batch = np.zeros((batch_size, size, size))
    t_batch = np.zeros((batch_size, size, size))

    nod_batch_2 = np.zeros((batch_size, size, size))

    dsm_batch_2 = np.zeros((batch_size, size, size))
    t = 0

    for i in range(batch_num * batch_size, (batch_num + 1) * batch_size):
        x_batch_2[t, :, :, :] = misc.imread(data_files[i, 0], mode='RGB')
        y_batch[t, :, :] = misc.imread(data_files[i, 1], mode='RGB')[:,:,0]
        z_batch[t, :, :] = misc.imread(data_files[i, 2], mode='RGB')[:,:,0]/255
        t_batch[t, :, :] = (misc.imread(data_files[i, 3], mode='RGB')[:,:,0]/255)*1.274
        nod_batch_2[t, :, :] = cv2.imread(data_files[i, 4], flags= cv2.IMREAD_UNCHANGED)
        dsm_batch_2[t, :, :] = cv2.imread(data_files[i, 5], flags= cv2.IMREAD_UNCHANGED)
        t += 1

    x_batch[:,:,:,0] = (x_batch_2[:,:,:,0] - mean[0])/255
    x_batch[:, :, :, 1] = (x_batch_2[:, :, :, 1] - mean[1])/255
    x_batch[:, :, :, 2] = (x_batch_2[:, :, :, 2] - mean[2])/255
    nod_batch = (nod_batch_2 -mean[3])/217.738355766
    dsm_batch = (dsm_batch_2/100.0 - mean[4])/74.132013621

    return x_batch, y_batch, z_batch, t_batch, nod_batch, dsm_batch

def supply_potsdam(batch_num, batch_size ,data_files,size,mean):


    x_batch = np.zeros((batch_size, size, size, 3))
    x_batch_2 = np.zeros((batch_size, size, size, 3))
    nod_batch_2 = np.zeros((batch_size, size, size))
    dsm_batch_2 = np.zeros((batch_size, size, size))
    t = 0

    for i in range(batch_num * batch_size, (batch_num + 1) * batch_size):
        x_batch_2[t, :, :, :] = misc.imresize(misc.imread(data_files[0], mode='RGB'),(512,512),interp='bicubic')
        nod_batch_2[t, :, :] = misc.imresize(np.load(data_files[1]),(512,512),interp='bicubic')
        dsm_batch_2[t, :, :] = misc.imresize(np.load(data_files[2]),(512,512),interp='bicubic')
        t += 1

    x_batch[:,:,:,0] = (x_batch_2[:,:,:,0] - mean[0])/255
    x_batch[:, :, :, 1] = (x_batch_2[:, :, :, 1] - mean[1])/255
    x_batch[:, :, :, 2] = (x_batch_2[:, :, :, 2] - mean[2])/255
    nod_batch = (nod_batch_2 -mean[3])/230
    dsm_batch = (dsm_batch_2 - mean[4])/71.073





    return x_batch, nod_batch, dsm_batch
