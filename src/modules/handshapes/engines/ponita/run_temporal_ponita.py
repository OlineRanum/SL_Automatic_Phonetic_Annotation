

import argparse
import os
from functools import partial

import wandb
import torch
import pytorch_lightning as pl
from PoseTools.src.models.temporal_ponita.lightning_wrappers.callbacks import EMA, EpochTimer
from PoseTools.src.models.temporal_ponita.lightning_wrappers.isr import PONITA_ISR

from PoseTools.src.models.temporal_ponita.datasets.isr.pyg_dataloader_isr import ISRDataLoader, ISRDataReader



# TODO: do we need this?
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def train(args):
    ########
    if args.gpus > 0:
        accelerator = "gpu"
        devices = args.gpus
    else:
        accelerator = "cpu"
        devices = "auto"
    if args.num_workers == -1:
        args.num_workers = os.cpu_count()

    # ------------------------ Dataset Loader
    
    # Load the dataset
    data_dir = os.path.dirname(__file__) + '/' + args.root
    data = ISRDataReader(data_dir, args)

    # Dataloader
    pyg_loader = ISRDataLoader(data, args)
    
    
    # ------------------------ Load and initialize the model
    model = PONITA_ISR(args)
        # After creating the model
    #dummy_input = torch.randn(args.batch_size, 64).to(devices)  # Adjust batch_size and input_shape
    #model(dummy_input)  # This runs a dummy forward pass to initialize the parameters


    # ------------------------ Weights and Biases logger
    logger = pl.loggers.WandbLogger(project=args.wandb_log_folder, name=args.model_name, config=args, save_dir=args.wandb_log_folder)

    # ------------------------ Set up the trainer
    
    # Seed
    pl.seed_everything(args.seed, workers=True)
    
    # Pytorch lightning call backs
    callbacks = [EMA(0.99),
                 pl.callbacks.ModelCheckpoint(monitor='val_acc', mode = 'max'),
                 EpochTimer()]
    if args.log: callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    
    # Initialize the trainer
    # trainer = pl.Trainer(gpus = 1, logger=logger, max_epochs=args.epochs, callbacks=callbacks, inference_mode=False, 
    #                     gradient_clip_val=0.5, accelerator=accelerator, devices=devices, enable_progress_bar=args.enable_progress_bar,
    #                     resume_from_checkpoint=args.checkpoint_path)
    trainer = pl.Trainer(gpus = args.gpus, logger=logger, max_epochs=args.epochs, callbacks=callbacks, inference_mode=False, 
                        gradient_clip_val=0.5, accelerator=accelerator, devices=devices, enable_progress_bar=args.enable_progress_bar)

#    trainer = pl.Trainer(gpus = 1, logger=logger, max_epochs=args.epochs, callbacks=callbacks, inference_mode=False, # Important for force computation via backprop
#                         gradient_clip_val=0.5, accelerator=accelerator, devices=devices, enable_progress_bar=args.enable_progress_bar,
#                          resume_from_checkpoint=args.resume_from_checkpoint)

    # Do the training
    trainer.fit(model, pyg_loader.train_loader, pyg_loader.val_loader)
    
    # And test
    single_view_mode = True
    if single_view_mode:
        trainer.test(model, pyg_loader.test_loader, ckpt_path = "best")
    else:
        trainer.test(model, pyg_loader.test_loader[0] , ckpt_path = "best")
        model.dataloader_idx = 1
        trainer.test(model, pyg_loader.test_loader[1] , ckpt_path = "best")
        model.dataloader_idx = 2
        trainer.test(model, pyg_loader.test_loader[2] , ckpt_path = "best")
    
    # Finish wandb to get different wandb runs
    wandb.run.finish()

    return model.top_val_metric






if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------------ Input arguments


    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to a checkpoint file to resume training')

    
    # Run parameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--warmup', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-10,
                        help='weight decay')
    parser.add_argument('--temporal_weight_decay', type=float, default=1e-3,
                        help='weight decay on parameters in 1D temporal conv')
    parser.add_argument('--temporal_dropout_rate', type=float, default=0.1,
                        help='dropout rate on parameters in 1D temporal conv')
    parser.add_argument('--log', type=eval, default=True,
                        help='logging flag')
    parser.add_argument('--model_name', type=str, default='Ponita_5c_200_sb',
                        help='logging flag')
    parser.add_argument('--wandb_log_folder', type=str, default='slgcn_sb',
                        help='logging flag')
    parser.add_argument('--enable_progress_bar', type=eval, default=True,
                        help='enable progress bar')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='Num workers in dataloader')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    # Settings for saving the model
    parser.add_argument('--save_folder', type=str, default='./logs/',
                        help='logging flag')
    
    # Train settings
    parser.add_argument('--train_augm', type=eval, default=False,
                        help='whether or not to use random rotations during training')
        
    # Graph connectivity settings
    parser.add_argument('--radius', type=eval, default=None,
                        help='radius for the radius graph construction in front of the force loss')
    parser.add_argument('--loop', type=eval, default=True,
                        help='enable self interactions')

    # PONTA model settings
    parser.add_argument('--num_ori', type=int, default=1,
                        help='num elements of spherical grid')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='internal feature dimension')
    parser.add_argument('--basis_dim', type=int, default=64,
                        help='number of basis functions')
    parser.add_argument('--degree', type=int, default=1,
                        help='degree of the polynomial embedding')
    parser.add_argument('--layers', type=int, default=6,
                        help='Number of message passing layers')
    parser.add_argument('--widening_factor', type=int, default=4,
                        help='Number of message passing layers')
    parser.add_argument('--layer_scale', type=float, default=0,
                        help='Initial layer scale factor in ConvNextBlock, 0 means do not use layer scale')
    parser.add_argument('--multiple_readouts', type=eval, default=False,
                        help='Whether or not to readout after every layer')
    # TIME PONITA model spesific settings
    parser.add_argument('--kernel_size', type=int, default=9,
                        help='size of 1D conv kernel')    
    parser.add_argument('--stride', type=int, default=1,
                        help='size of 1D conv stride')    
    
    # ISR Dataset settings
    parser.add_argument('--root', type=str, default="",
                        help='Data set location')
    parser.add_argument('--root_metadata', type=str,  default="PoseTools/data/metadata/output/5c_sb/5c_sb.json",
                        help='Metadata json file location')
    import os

    parser.add_argument('--root_poses', type=str, default=os.path.abspath("../../../../mnt/fishbowl/gomer/oline/hamer_pkl"),
                    help='Pose data dir location')
    #parser.add_argument('--root_poses', type=str, default="../../../../mnt/fishbowl/gomer/oline/hamer_pkl",
    #                    help='Pose data dir location')
    parser.add_argument('--n_classes', type=str, default=5,
                        help='Number of sign classes')
    parser.add_argument('--n_targets', type=str, default=1,
                        help='Number of attribute classes')
    parser.add_argument('--temporal_configuration', type=str, default="spatio_temporal",
                        help='Temporal configuration of the graph. Options: spatio_temporal, per_frame') 
    parser.add_argument('--n_nodes', type=int, default=21,
                        help='Number of nodes to use when reducing the graph - only 27 currently implemented')
    parser.add_argument('--scale_norm', type=eval, default=True,
                        help='If to apply scale and normalization') 
    parser.add_argument('--downsample', type=eval, default=False,
                        help='If to apply scale and normalization')
        
    # Parallel computing stuff
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus to use (assumes all are on one node)')
    
    args = parser.parse_args()
    
    train(args)