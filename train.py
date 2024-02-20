import time
import argparse
import torch
from model import EnhanceGANModel
from dataset import AlignedDataset
from util import Visualizer

def get_parser():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    # Dataset path and model options
    parser.add_argument('--data_root', default=r'E:\VEUS-main\VEUS-main', type=str, help='path/to/data')
    parser.add_argument('--EnhanceT', default=True, help='')

    # Training related arguments
    parser.add_argument('--isTrain', default=True, help='')
    parser.add_argument('--epoch_count', default=1, help='the starting epoch count')
    parser.add_argument('--lr', default=0.0002, help='learning rate')
    parser.add_argument('--lr_policy', default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--batch_size', default=1, help='')
    parser.add_argument('--niter', default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--epoch', default='latest', help='')
    parser.add_argument('--gpu_ids', default=[0], help='which device')

    # Checkpoint and experiment naming
    parser.add_argument('--checkpoints_dir', default='./checkpoints', help='')
    parser.add_argument('--name', default='experiment_name', help='')
    return parser.parse_args()

if __name__ == '__main__':
    opt = get_parser()

    # Initialize dataset and DataLoader
    dataset = AlignedDataset(opt.data_root, 'train')
    dataset_size = len(dataset)
    dataset = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not False,
        num_workers=0)
    print('The number of training images = %d' % dataset_size)

    # Initialize the model, visualizer, and setup
    model = EnhanceGANModel(opt)
    model.setup(opt)

    visualizer = Visualizer(opt)
    total_iters = 0

    # Training loop
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % 100 == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            # Visualization and logging
            if total_iters % 400 == 0:
                save_result = total_iters % 1000 == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # Print current losses
            if total_iters % 100 == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            # Saving the model periodically
            if total_iters % 5000 == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        # Saving the model at the end of every few epochs
        if epoch % 5 == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # Print epoch end summary
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, 100 + 100, time.time() - epoch_start_time))
        # Update learning rate
        model.update_learning_rate()
