'''
Train and use learned models for QC of cell tracks
'''
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import os.path as osp
import glob
from typing import Union, Callable
from torchvision.models import squeezenet1_1

from sklearn.model_selection import StratifiedKFold
from qc_model_trainer import Trainer
from qc_dataprep import TrackParser, TrackDataset, RandomNoise

def mkdir_f(d):
    if not osp.exists(d):
        os.mkdir(d)

class TrackClassifier(nn.Module):

    def __init__(self,
        n_dimensions: int=2,
        n_classes: int=2,
        n_hidden: int=1024,
        n_features: int=6,
        rnn_unit: str='gru',
        use_images: bool=False,
        dropout_p: float=0.3) -> None:
        '''Classification model for tracking inputs

        Parameters
        ----------
        n_dimensions : int
            number of input dimensions.
        n_classes : int
            number of output classes.
        n_hidden : int
            hidden RNN units.
        n_features : int
            number of heuristic features.
        rnn_unit : str
            {'gru', None} type of RNN unit.
            if None, doesn't use an RNN at all and relies only on features.
        dropout_p : float
            proportion of nodes to drop in Dropout layers.

        Returns
        -------
        None.
        '''

        super(TrackClassifier, self).__init__()
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.n_features = n_features
        self.rnn_unit = rnn_unit
        self.dropout_p = dropout_p

        if rnn_unit is None:
            pass
        elif rnn_unit.lower() == 'gru':
            self.rnn = nn.Sequential(
                nn.GRU(input_size=self.n_dimensions,
                              hidden_size=self.n_hidden,
                              num_layers=2,
                              batch_first=True),
            )

        else:
            raise NotImplementedError()

        if rnn_unit is not None:
            rnn_out_units = 256
            self.rnn_fc = nn.Sequential(
                nn.Linear(self.n_hidden, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_p),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_p),
                nn.Linear(256, rnn_out_units),
                nn.ReLU(inplace=True),
            )
        else:
            rnn_out_units = 0

        feat_out_units = 16
        self.feat_fc = nn.Sequential(
            nn.Linear(self.n_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, feat_out_units),
            nn.ReLU(inplace=True),
        )

        if self.use_images:
            im_out_units = 256
            self.sq = squeezenet1_1(pretrained=True)
            self.im_conv = nn.Sequential(
                self.sq,
                nn.Linear(1000, im_out_units),
                nn.Dropout(self.dropout_p),
                nn.ReLU(inplace=True),
            )
        else:
            im_out_units = 0

        self.clf = nn.Sequential(
            nn.Linear(rnn_out_units + feat_out_units + im_out_units,
                      256),
            nn.ReLU(inplace=True)
            nn.Linear(256, self.n_classes),
            nn.ReLU(inplace=True),
        )

    def forward(self,
                x_track: torch.FloatTensor,
                x_features: torch.FloatTensor,
                x_img: torch.FloatTensor=None,) -> torch.FloatTensor:
        '''
        Parameters
        ----------
        x_track : torch.FloatTensor
            [N, Time, Dimensions]
        x_features : torch.FloatTensor
            [N, Features]
        x_img : torch.FloatTensor
            [N, C, H, W] images.

        Returns
        -------
        scores : torch.FloatTensor
            [N, n_classes]
        '''
        feat_scores = self.feat_fc(x_features)
        if self.rnn_unit is not None:
            # outputs: [batch, seq_len, last_hidden_size]
            outputs, hidden = self.rnn(x_track) # [N, T, D]
            rnn_vec = outputs[:, -1, :] # [Batch, n_hidden]
            rnn_scores = self.rnn_fc(rnn_vec) # [Batch, 256]

            clf_vec = torch.cat([rnn_scores, feat_scores], dim=1)
        else:
            clf_vec = feat_scores

        if x_img is not None:
            img_vec = self.im_conv(x_img)
            clf_vec = torch.cat([clf_vec, img_vec])

        scores = self.clf(clf_vec)
        return scores

def train_rnn_model_cv(params: dict,):

    mkdir_f(params.get('out_path'))

    tp = TrackParser(params.get('tracks_dir'),
                     params.get('all_tracks_glob'),
                     params.get('kept_tracks_glob'))
    N = tp.tracks.size(0)
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    kf_indices = list(kf.split(tp.tracks.numpy(),
                               tp.labels.numpy()))
    fold_eval_losses = np.zeros(5)
    fold_eval_acc = np.zeros(5)
    fold_eval_fp = np.zeros(5)
    fold_baselines = np.zeros((5, 2)) # [folds, (acc, false_pos)]

    for f in range(5):

        fold_out_path = osp.join(params.get('out_path'),
                                'fold' + str(f).zfill(2))
        mkdir_f(fold_out_path)

        train_idx = kf_indices[f][0].astype('int')
        val_idx = kf_indices[f][1].astype('int')

        train_track_ds = TrackDataset(tp.tracks[train_idx, :, :],
                                tp.labels[train_idx],
                                do_class_balancing=params.get('do_class_balancing', False),
                                center_tracks=params.get('center_tracks', False),
                                use_features=params.get('use_features', True),
                                transform=RandomNoise(sigma=1.))
        val_track_ds = TrackDataset(tp.tracks[val_idx, :, :],
                              tp.labels[val_idx],
                              do_class_balancing=params.get('do_class_balancing', False),
                              center_tracks=params.get('center_tracks', False),
                              use_features=params.get('use_features', True))

        if params.get('use_images', False):
            train_ds = TrackImageDataset(track_ds=train_track_ds,
                                        img_dir=params['img_dir'],
                                        img_glob=params['img_glob'],
                                        im_transform=imgnet_trans,
                                        bbox_sz=(128,128))
            val_ds = TrackImageDataset(track_ds=val_track_ds,
                                       img_dir=params['img_dir'],
                                       img_glob=params['img_glob'],
                                       im_transform=imgnet_trans,
                                       bbox_sz=(128,128))
        else:
            train_ds = train_track_ds
            val_ds = val_track_ds

        print('BASELINE ACCURACY')
        classes, counts = np.unique(train_ds.labels.numpy(), return_counts=True)
        most_common = classes[np.argmax(counts)]
        print('Most common class: ', most_common)
        print('Training:')
        train_baseline = (train_ds.labels.numpy() == most_common).sum() / len(train_ds)
        print(train_baseline)
        print('Testing:')
        test_baseline = (val_ds.labels.numpy() == most_common).sum() / len(val_ds)
        print(test_baseline)

        train_dl = DataLoader(train_ds,
            batch_size=params.get('batch_size'), shuffle=True)
        val_dl = DataLoader(val_ds,
            batch_size=params.get('batch_size'), shuffle=False)
        dataloaders = {'train':train_dl,
                       'val':val_dl}

        model = TrackClassifier(n_dimensions=tp.tracks.size(2),
                                n_classes=2,
                                rnn_unit=params.get('rnn_unit', 'gru'),
                                use_images=params.get('use_images', False))

        if torch.cuda.is_available():
            model = model.cuda()

        optimizer = torch.optim.Adadelta(model.parameters(),
                                    lr=params.get('lr', 0.1),
                                    weight_decay=params.get('weight_decay', 0.0))
        if params.get('use_images', False):
            model.sq.eval()

        if params.get('class_weights', False):
            p_discard = float(np.sum(tp.labels.numpy() == 0)/tp.labels.shape[0])
            print('Proportion of discarded examples: %f' % p_discard)
            class_weights = np.array([1./p_discard, 1.])
            class_weights = torch.from_numpy(class_weights).float()
            print('Class weights for training:')
            print(class_weights)
            if torch.cuda.is_available():
                class_weights = class_weights.cuda()
        else:
            class_weights = None

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        T = Trainer(model=model,
                    criterion=criterion,
                    dataloaders=dataloaders,
                    optimizer=optimizer,
                    out_path=fold_out_path,
                    n_epochs=params.get('n_epochs'),
                    use_images=params.get('use_images', False),
                    exp_name='track_qc_model',
                    early_stopping=None)
        T.train()

        print('Evaluating...')

        model = TrackClassifier(n_dimensions=tp.tracks.size(2),
                                n_classes=2,
                                rnn_unit=params.get('rnn_unit', None),
                                use_images=params.get('use_images', False))
        model.load_state_dict(
            torch.load(
                osp.join(fold_out_path,
                'model_weights_' + str(params.get('n_epochs')-1).zfill(4) + '.pkl')
                )
        )
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()

        with torch.no_grad():
            loss = 0.
            running_corrects = 0.
            running_total = 0.
            all_predictions = []
            all_labels = []
            for data in val_dl:
                input_ = data['input']
                features = data['features']
                label_ = data['output']

                if torch.cuda.is_available():
                    input_ = input_.cuda()
                    features = features.cuda()
                    label_ = label_.cuda()

                if params.get('use_images', False):
                    image = data['image']
                    if torch.cuda.is_available():
                        image = image.cuda()
                    image.requires_grad = False
                else:
                    image = None

                input_.requires_grad = False
                features.requires_grad = False
                label_.requires_grad = False

                output = model(input_, features, image)
                _, predictions = torch.max(output, 1)
                corrects = torch.sum(predictions.detach() == label_.detach())

                l = criterion(output, label_)
                loss += float(l.detach().cpu().numpy())

                running_corrects += float(corrects.item())
                running_total += float(label_.size(0))

                all_labels.append( label_.detach().cpu().numpy() )
                all_predictions.append( predictions.detach().cpu().numpy() )

            norm_loss = loss / len(val_dl)
            print('EVAL LOSS: ', norm_loss)
            print('EVAL ACC : ', running_corrects/running_total)
        fold_eval_acc[f] = running_corrects/running_total
        fold_eval_losses[f] = norm_loss

        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        np.savetxt(osp.join(fold_out_path, 'predictions.csv'), all_predictions)
        np.savetxt(osp.join(fold_out_path, 'labels.csv'), all_labels)

        PL = np.stack([all_predictions, all_labels], 0)
        classes, counts = np.unique(all_labels, return_counts=True)
        baseline_acc = counts.max() / counts.sum()
        pred_fp = np.logical_and(all_predictions==1,
                                    all_labels==0).sum() / len(all_predictions)
        fold_eval_fp[f] = pred_fp
        baseline_fp = np.sum(all_labels==0)/len(all_labels)
        fold_baselines[f, 0] = baseline_acc
        fold_baselines[f, 1] = baseline_fp
        print('Predictions | Labels')
        print(PL[:15,:])
        print('Baseline Accuracy')
        print(baseline_acc)
        print('Baseline False Positives')
        print(baseline_fp)
        print('Model Prediction False Positives')
        print(pred_fp)

    print('Fold eval losses')
    print(fold_eval_losses)
    print('Fold eval accuracy')
    print(fold_eval_acc)
    print('Fold eval fp')
    print(fold_eval_fp)
    print('Fold baseline accuracy')
    print(fold_baselines[:,0])
    print('Fold baseline fp')
    print(fold_baselines[:,1])
    print('Mean %f Std %f' % (fold_eval_losses.mean(), fold_eval_losses.std()))
    np.savetxt(osp.join(params.get('out_path'), 'fold_eval_losses.csv',), fold_eval_losses)
    np.savetxt(osp.join(params.get('out_path'), 'fold_eval_acc.csv',), fold_eval_acc)
    np.savetxt(osp.join(params.get('out_path'), 'fold_eval_fp.csv',), fold_eval_fp)
    np.savetxt(osp.join(params.get('out_path'), 'fold_baselines.csv',), fold_baselines)
    return

def predict_qc(params: dict,) -> np.ndarray:

    tp = TrackParser(params.get('tracks_dir'),
                     params.get('all_tracks_glob'),
                     kept_tracks_glob=None)

    pred_ds = TrackDataset(tp.tracks,
                            tp.labels,
                            do_class_balancing=False,
                            center_tracks=params.get('center_tracks', False),
                            use_features=params.get('use_features', True))

    pred_dl = DataLoader(pred_ds, batch_size=params['batch_size'], shuffle=False)

    model = TrackClassifier(n_dimensions=tp.tracks.size(2),
                            n_classes=2,
                            rnn_unit=params['rnn_unit'])

    model.load_state_dict(
        torch.load(params['model_weights'], map_location='cpu'))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    with torch.no_grad():
        all_predictions = []

        for data in pred_dl:
            input_ = data['input']
            features = data['features']

            if torch.cuda.is_available():
                input_ = input_.cuda()
                features = features.cuda()

            input_.requires_grad = False
            features.requires_grad = False

            output = model(input_, features)
            _, predictions = torch.max(output, 1)

            all_predictions.append( predictions.detach().cpu().numpy() )

        all_predictions = np.concatenate(all_predictions)
        np.savetxt(osp.join(params['out_path'], 'predictions.csv'),
            all_predictions, delimiter=',')

        print('Proportion of tracks kept:')
        print(np.sum(all_predictions == 1)/len(all_predictions))
        print('Proportion of tracks discarded:')
        print(np.sum(all_predictions == 0)/len(all_predictions))

    # save new tracks after cleaning
    tp.clean_track_files(all_predictions, save_suffix='_qc_model')

    return all_predictions

def main():
    import configargparse
    parser = configargparse.ArgParser(description='Train tracking QC models',
            default_config_files=['./track_qc_config.txt'])
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('command', type=str,
        help='action to perform: [train,]')
    parser.add_argument('--tracks_dir', type=str, help='dir containing tracks')
    parser.add_argument('--all_tracks_glob', type=str,
            default='*densenet*tracks*.csv',
            help='pattern to match filenames of raw, non-QCed tracks')
    parser.add_argument('--kept_tracks_glob', type=str,
            default='*densenet*tracks*pruned.csv',
            help='pattern to match filenames of manually QCed tracks derived \
            from corresponding `all_tracks_glob` files')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4,
        help='l2 regularization strength')
    parser.add_argument('--out_path', type=str, default='./')
    parser.add_argument('--do_class_balancing', action='store_true')
    parser.add_argument('--class_weights', action='store_true')
    parser.add_argument('--use_features', action='store_true',
        help='use heuristic features in addition to raw tracks')
    parser.add_argument('--use_images', action='store_true',
        help='use images in addition to tracks')
    parser.add_argument('--img_dir', type=str, default=None,
        help='path to image files')
    parser.add_argument('--img_glob', type=str, default='*.tif',
        help='path to match image files')
    parser.add_argument('--rnn_unit', type=str, default='gru',
        help='RNN unit to use')
    parser.add_argument('--model_weights', type=str, default=None,
        help='model weights to use for predictions')
    args = parser.parse_args()
    print(args)
    print(parser.format_values())

    params = {
        'out_path': args.out_path,
        'tracks_dir': args.tracks_dir,
        'all_tracks_glob': args.all_tracks_glob,
        'kept_tracks_glob': args.kept_tracks_glob,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'do_class_balancing': args.do_class_balancing,
        'class_weights': args.class_weights,
        'lr': args.lr,
        'use_features': args.use_features,
        'center_tracks': True,
        'model_weights': args.model_weights,
        'rnn_unit': None if args.rnn_unit == 'None' else args.rnn_unit,
        'weight_decay': args.weight_decay,
        'img_dir': args.img_dir,
        'img_glob': args.img_glob,
        'use_images': args.use_images,
    }

    if args.command.lower() == 'train':
        train_rnn_model_cv(params)

    if args.command.lower() == 'predict':
        if params['model_weights'] is None:
            raise ValueError('must supply model weights for predictions!')
        predict_qc(params)

if __name__ == '__main__':
    main()
