import numpy as np
import pandas as pd
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
import sys
import json
from typing import Union

class Trainer(object):
    '''
    Trains a model
    '''

    def __init__(self,
                model: nn.Module,
                criterion: Callable,
                optimizer,
                dataloaders: dict,
                out_path: str,
                n_epochs: int=50,
                exp_name: str='',
                use_gpu: bool=torch.cuda.is_available(),
                use_images: bool=False,
                verbose: bool=False,
                save_freq: int=10,
                scheduler = None,
                early_stopping: int=None):

        '''
        Trains a PyTorch `nn.Module` object provided in `model`
        on training and testing sets provided in `dataloaders`
        using `criterion` and `optimizer`.

        Saves model weight snapshots every `save_freq` epochs and saves the
        weights with the best testing loss at the end of training.

        Parameters
        ----------
        model : torch model object
            must have a callable `forward` method.
        criterion : callable
            takes inputs and targets, returns loss.
        optimizer : torch.optim optimizer.
        dataloaders : dict
            train, val dataloaders keyed 'train', 'val'.
        out_path : str
            output path for best model.
        n_epochs : int
            number of epochs for training.
        use_gpu : bool
            use CUDA acceleration.
        verbose : bool
            write all batch losses to stdout.
        save_freq : int
            Number of epochs between model checkpoints. Default = 10.
        scheduler : learning rate scheduler.
        early_stopping : int
            if None, no early stopping. Otherwise, number of epochs to wait
            before stopping.
        '''
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.dataloaders = dataloaders
        self.out_path = out_path
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.save_freq = save_freq
        self.best_acc = 0.
        self.best_loss = 1.0e10
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.pr_thresholds = [0.0, 0.3, 0.5, 0.7]

        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        # initialize log

        self.log_path = os.path.join(self.out_path, '_'.join([exp_name, 'log.csv']))
        with open(self.log_path, 'w') as f:
            header = 'Epoch,Running_Loss,Mode\n'
            f.write(header)

        self.parameters = {
            'out_path': out_path,
            'exp_name': exp_name,
            'n_epochs': n_epochs,
            'use_cuda': self.use_gpu,
            'train_batch_size': self.dataloaders['train'].batch_size,
            'val_batch_size': self.dataloaders['val'].batch_size,
            'train_batch_sampler': type(self.dataloaders['train'].sampler),
            'val_batch_sampler': type(self.dataloaders['val'].sampler),
            'optimizer_type': type(self.optimizer),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'model_hidden': self.model.n_hidden,
        }

        # # save parameters to JSON
        # json.dump(self.parameters,
        #          open(osp.join(self.out_path, exp_name + '_parameters.json'), 'w'))

        with open(self.log_path, 'w') as f:
            header = 'Epoch,Iter,Running_Loss,Mode\n'
            f.write(header)

    def train_epoch(self):
        # Run a train and validation phase for each epoch
        self.model.train(True)
        i = 0
        running_loss = 0.0
        running_corrects = 0.
        running_total = 0.
        for data in self.dataloaders['train']:
            inputs, labels = data['input'], data['output']
            features = data['features']
            if self.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
                features = features.cuda()
            else:
                pass
            inputs.requires_grad = True
            features.requires_grad = True
            labels.requires_grad = False

            if self.use_images:
                image = data['image']
                if self.use_gpu:
                    image = image.cuda()
                image.requires_grad = True
            else:
                image = None

            # zero gradients
            self.optimizer.zero_grad()

            # forward pass
            outputs = self.model(inputs, features, image)

            _, predictions = torch.max(outputs, 1)

            if self.verbose:
                print('Preds:')
                print(predictions[:10])
                print('Labels:')
                print(labels[:10])

            correct = torch.sum(predictions.detach() == labels.detach())
            acc = correct / labels.size(0)

            loss = self.criterion(outputs, labels)
            if self.verbose:
                print('batch loss: ', loss.item())
            assert np.isnan(loss.data.cpu().numpy()) == False, 'NaN loss encountered in training'

            # backward pass
            loss.backward()
            self.optimizer.step()

            # statistics update
            running_loss += loss.item() / inputs.size(0)
            running_corrects += float(correct.item())
            running_total += float(labels.size(0))

            if i % 100 == 0:
                print('Iter : ', i)
                print('running_loss : ', running_loss / (i + 1))
                print('running_acc  : ', running_corrects/running_total)
                print('corrects: %f | total: %f' % (running_corrects, running_total))
                # append to log
                with open(self.log_path, 'a') as f:
                    f.write(str(self.epoch) + ',' + str(i) + ',' + str(running_loss / (i + 1)) + ',train\n')
            i += 1

        epoch_loss = running_loss / len(self.dataloaders['train'])
        epoch_acc  = running_corrects / running_total

        # append to log
        with open(self.log_path, 'a') as f:
            f.write(str(self.epoch) + ',' + str(i) + ',' + str(running_loss / (i + 1)) + ',train_epoch\n')

        print('{} Loss : {:.4f}'.format('train', epoch_loss))
        print('{} Acc : {:.4f}'.format('train', epoch_acc))
        print('TRAIN EPOCH corrects: %f | total: %f' % (running_corrects, running_total))

    def val_epoch(self):
        self.model.train(False)
        i = 0
        running_loss = 0.0
        running_corrects = 0
        running_total = 0
        running_pr = np.zeros((2,2,len(self.pr_thresholds)))
        for data in self.dataloaders['val']:
            inputs, labels = data['input'], data['output']
            features = data['features']
            if self.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
                features = features.cuda()
            else:
                pass
            inputs.requires_grad = False
            features.requires_grad = False
            labels.requires_grad = False # just double check these are volatile

            if self.use_images:
                image = data['image']
                if self.use_gpu:
                    image = image.cuda()
                image.requires_grad = False
            else:
                image = None

            # zero gradients
            self.optimizer.zero_grad()
            # forward pass
            outputs = self.model(inputs, features, image)
            _, predictions = torch.max(outputs, 1)
            correct = torch.sum(predictions.detach() == labels.detach())
            print('PRED\n', predictions[:10,...])
            print('LABEL\n', labels[:10,...])
            print('CORRECT: ', correct)
            acc = correct / labels.size(0)

            pr = self.precision_recall(labels.cpu(), F.softmax(outputs).cpu())

            loss = self.criterion(outputs, labels)

            # statistics update
            running_loss += loss.item() / inputs.size(0)
            running_corrects += int(correct.item())
            running_total += int(labels.size(0))
            running_pr += pr

            if i % 1 == 10:
                print('Iter : ', i)
                print('running_loss : ', running_loss / (i + 1))
                print('running_acc  : ', running_corrects/running_total)
                print('corrects: %f | total: %f' % (running_corrects, running_total))
                # append to log
                with open(self.log_path, 'a') as f:
                    f.write(str(self.epoch) + ',' + str(i) + ',' + str(running_loss / (i + 1)) + ',val\n')
            i += 1

        epoch_loss = running_loss / len(self.dataloaders['val'])
        epoch_acc  = running_corrects / running_total
        # append to log
        with open(self.log_path, 'a') as f:
            f.write(str(self.epoch) + ',' + str(i) + ',' + str(running_loss / (i + 1)) + ',val_epoch\n')

        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.best_model_wts = self.model.state_dict()
            torch.save(self.model.state_dict(), os.path.join(self.out_path, 'model_weights_' + str(self.epoch).zfill(4) + '.pkl'))
        elif (self.epoch%self.save_freq == 0) or (self.epoch==self.n_epochs-1):
            torch.save(self.model.state_dict(),
                       os.path.join(self.out_path, 'model_weights_' + str(self.epoch).zfill(4) + '.pkl'))

        print('PRECISION:RECALL\n')
        for i in range(running_pr.shape[2]):
            print(running_pr[:,:,i]/running_pr[:,:,i].sum())
            print()

        print('{} Loss : {:.4f}'.format('val', epoch_loss))
        print('{} Acc : {:.4f}'.format('val', epoch_acc))
        print('VAL EPOCH corrects: %f | total: %f' % (running_corrects, running_total))
        return epoch_loss

    def precision_recall(self,
                    labels,
                    outputs) -> np.ndarray:
        '''Generate precision:recall numbers for a binary classification problem

        Parameters
        ----------
        labels : torch.LongTensor
        outputs : torch.FloatTensor
            [Batch, (Negative, Positive)] softmax class probabilities.

        Returns
        -------
        pr : np.ndarray
            Precision recall matrix [2, 2, len(thresholds)]
            [ [tp, fp], [tn, fn] ]
        '''
        gt_pos = labels.detach().numpy() == 1.
        gt_neg = ~gt_pos

        pr = np.zeros((2, 2, len(self.pr_thresholds)))

        for i, t in enumerate(self.pr_thresholds):
            predictions = outputs.detach().numpy()[:,1] > t

            pred_pos = predictions == 1.
            pred_neg = ~pred_pos

            true_pos = np.logical_and(gt_pos, pred_pos).sum()
            true_neg = np.logical_and(gt_neg, pred_neg).sum()
            false_pos = np.logical_and(gt_neg, pred_pos).sum()
            false_neg = np.logical_and(gt_pos, pred_neg).sum()

            pr[:,:,i] = np.array([
                [true_pos, false_pos],
                [true_neg, false_neg],
                ])
        return pr

    def train(self):

        epochs_since_best = 0
        best_loss = 1e6
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            print('Epoch {}/{}'.format(epoch, self.n_epochs - 1))
            print('-' * 10)
            # run training epoch
            if self.scheduler is not None:
                self.scheduler.step()
            self.train_epoch()
            with torch.no_grad():
                val_loss = self.val_epoch()

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            if self.early_stopping is not None:
                if epochs_since_best > self.early_stopping:
                    break

        print('Saving best model weights...')
        torch.save(self.best_model_wts, os.path.join(self.out_path, '00_best_model_weights.pkl'))
        print('Saved best weights.')
        self.model.load_state_dict(self.best_model_wts)
        return self.model
