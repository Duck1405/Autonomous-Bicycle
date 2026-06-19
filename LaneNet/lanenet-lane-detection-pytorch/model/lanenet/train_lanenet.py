import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import copy
import os
from tqdm import tqdm
from model.lanenet.loss import DiscriminativeLoss, FocalLoss

def compute_loss(net_output, binary_label, instance_label, loss_type = 'FocalLoss'):
    k_binary = 10    #1.7
    k_instance = 0.3
    k_dist = 1.0

    if(loss_type == 'FocalLoss'):
        loss_fn = FocalLoss(gamma=2, alpha=[0.25, 0.75])
    elif(loss_type == 'CrossEntropyLoss'):
        loss_fn = nn.CrossEntropyLoss()
    else:
        # print("Wrong loss type, will use the default CrossEntropyLoss")
        loss_fn = nn.CrossEntropyLoss()
    
    binary_seg_logits = net_output["binary_seg_logits"]
    binary_loss = loss_fn(binary_seg_logits, binary_label)

    pix_embedding = net_output["instance_seg_logits"]
    ds_loss_fn = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)
    var_loss, dist_loss, reg_loss = ds_loss_fn(pix_embedding, instance_label)
    binary_loss = binary_loss * k_binary
    var_loss = var_loss * k_instance
    dist_loss = dist_loss * k_dist
    instance_loss = var_loss + dist_loss
    total_loss = binary_loss + instance_loss
    out = net_output["binary_seg_pred"]

    return total_loss, binary_loss, instance_loss, out


def train_model(model, optimizer, scheduler, dataloaders, dataset_sizes, device, loss_type = 'FocalLoss', num_epochs=25, save_path=None):
    since = time.time()
    training_log = {'epoch':[], 'training_loss':[], 'val_loss':[]}
    best_loss = float("inf")

    best_model_wts = copy.deepcopy(model.state_dict())
    print(f'Device: {device} in Train Model')
    print("Starting training loop")
    print("Loss type: {}".format(loss_type))
    print("Dataset sizes: train={} val={}".format(dataset_sizes['train'], dataset_sizes['val']))
    if save_path is not None:
        print("Saving epoch checkpoints to: {}".format(save_path))

    for epoch in range(num_epochs):
        training_log['epoch'].append(epoch)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_loss_b = 0.0
            running_loss_i = 0.0

            print("Starting {} phase with {} batches".format(phase, len(dataloaders[phase])))

            # Iterate over data.
            progress_bar = tqdm(dataloaders[phase], desc="{} epoch {}".format(phase, epoch), leave=False)
            samples_seen = 0
            for inputs, binarys, instances in progress_bar:
                batch_size = inputs.size(0)
                inputs = inputs.type(torch.FloatTensor).to(device)
                binarys = binarys.type(torch.LongTensor).to(device)
                instances = instances.type(torch.FloatTensor).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = compute_loss(outputs, binarys, instances, loss_type)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss[0].backward()
                        optimizer.step()

                # statistics
                running_loss += loss[0].item() * inputs.size(0)
                running_loss_b += loss[1].item() * inputs.size(0)
                running_loss_i += loss[2].item() * inputs.size(0)
                samples_seen += batch_size
                progress_bar.set_postfix({
                    'loss': '{:.4f}'.format(running_loss / max(samples_seen, 1)),
                    'binary': '{:.4f}'.format(running_loss_b / max(samples_seen, 1)),
                    'instance': '{:.4f}'.format(running_loss_i / max(samples_seen, 1)),
                })

            if phase == 'train':
                if scheduler != None:
                    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            binary_loss = running_loss_b / dataset_sizes[phase]
            instance_loss = running_loss_i / dataset_sizes[phase]
            print('{} Total Loss: {:.4f} Binary Loss: {:.4f} Instance Loss: {:.4f}'.format(phase, epoch_loss, binary_loss, instance_loss))

            # deep copy the model
            if phase == 'train':
                training_log['training_loss'].append(epoch_loss)
            if phase == 'val':
                training_log['val_loss'].append(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

        if save_path is not None:
            epoch_save_filename = os.path.join(save_path, 'epoch_{:03d}.pth'.format(epoch + 1))
            torch.save(model.state_dict(), epoch_save_filename)
            print("epoch checkpoint is saved: {}".format(epoch_save_filename))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val_loss: {:4f}'.format(best_loss))
    training_log['training_loss'] = np.array(training_log['training_loss'])
    training_log['val_loss'] = np.array(training_log['val_loss'])

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_log

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
