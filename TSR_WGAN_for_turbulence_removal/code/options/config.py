import argparse

def train_options():
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument('--gpu_index', default='0', type=str, help='gpu ids:e.g. 0,1,2,...')
    parser.add_argument('--name', default='experiment_name', type=str, help='name of the experiment which corresponds '
                                                                            'to the store path')
    parser.add_argument('--checkpoint_dir', default='../data/checkpoints1', type=str, help='path to the saved model')
    parser.add_argument('--continue_train', action='store_true', help='continue training from the latest model')
    parser.add_argument('--start_epoch', default=1, type=int, help='the start epoch count for training')
    parser.add_argument('--n_epochs', default=100, type=int, help='number of epochs to be trained')
    parser.add_argument('--is_train', default=True, type=bool, help='true for traning and false for testing')
    parser.add_argument('--print_freq', default=15, type=int, help='frequency of showing training results on console')
    # training parameters
    parser.add_argument('--lr_g', default=3e-4, type=float, help='initial learning rate of generator')
    parser.add_argument('--lr_d', default=1e-5, type=float, help='initial learning rate of discriminator')
    parser.add_argument('--lr_scheduler', default='Step', type=str,
                        help='method to adjust the learning rate: RP for ReduceLROnPlateau and Step for StepLR')
    parser.add_argument('--lr_decay', default=0.7, type=float, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--lr_step_size', default=20, type=int, help='period of learning rate decay if StepLR '
                                                                     'scheduler is chosen')
    parser.add_argument('--time_length', default=15, type=int,help='time length for input sequence')
    parser.add_argument('--dis_length', default=15, type=int,help='length for discriminator input')
    parser.add_argument('--alpha', default=10, type=float,help='weight for perceptual loss')
    parser.add_argument('--beta', default=10000, type=float, help='weight for original mse loss')
    # dataset parameters
    parser.add_argument('--data_root', default='./data/train',type=str, help='path to images')
    parser.add_argument('--batch_size', default=3, type=int, help='batch size for training')
    parser.add_argument('--shuffle', default=True, type=bool, help='shuffle dataset or not')
    parser.add_argument('--val_path', default='./validate', type=str, help='path to validation data')
    parser.add_argument('--crop', default=True, type=bool, help='whether to crop image to small size')
    parser.add_argument('--crop_size', default=256, type=int, help='then crop to this size')
    parser.add_argument('--padding', default=False, type=bool, help='padding for integer multiple of 8')
    return parser

def test_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../data/test', type=str, help='path to the test data')
    parser.add_argument('--model_path', default='../data/checkpoints/experiment_name',type=str, help='path to the test model')
    parser.add_argument('--model_name', default='model_test.pth', type=str, help='name of the test model')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size for testing')
    parser.add_argument('--save_path', default='../results/test_results', help='path to the saved results')
    parser.add_argument('--gpu_index', default='0', type=str, help='gpu ids:e.g. 0,1,2,...')
    parser.add_argument('--is_train', default=False, type=bool, help='true for training and false for testing')

    parser.add_argument('--time_length', default=15, type=int, help='time length for input sequence')
    parser.add_argument('--dis_length', default=15, type=int, help='length for discriminator input')
    parser.add_argument('--shuffle', default=False, type=bool, help='shuffle dataset or not')
    parser.add_argument('--crop', default=False, type=bool, help='whether to crop image to small size')
    parser.add_argument('--crop_size', default=256, type=int, help='then crop to this size')
    parser.add_argument('--padding', default=True, type=bool, help='padding for integer multiple of 8')
    parser.add_argument('--duplicate', default=False, type=bool, help='duplicate frames to slow down the video')
    return parser