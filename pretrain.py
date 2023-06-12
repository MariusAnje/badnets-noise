import argparse
import os
import pathlib
import re
import time
import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import build_poisoned_training_set, build_testset
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch
from models import BadNet
from utils import str2bool, get_dataset, get_model, prepare_model
from utils import MTrain, TCEval, TMEachEval, CEval, MEachEval, UpdateBN
from utils import copy_model, get_poision_datasets, get_bad
from bad_attack import BadAttack, PGD, FGSM, LM, binary_search_dist

parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=0, help='Batch size to split dataset, default: 64')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate of the model, default: 0.001')
parser.add_argument('--download', action='store_true', help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('--data_path', default='~/Private/data', help='Place to load dataset (default: ./dataset/)')
# parser.add_argument('--device', default='cpu', help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
# poison settings
parser.add_argument('--poisoning_rate', type=float, default=0.0, help='poisoning portion (float, range from 0 to 1, default: 0.0)')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')
parser.add_argument('--train_epoch', action='store', type=int, default=20,
            help='# of epochs of training')
parser.add_argument('--noise_epoch', action='store', type=int, default=100,
        help='# of epochs of noise validations')
parser.add_argument('--train_var', action='store', type=float, default=0.1,
        help='device variation [std] when training')
parser.add_argument('--dev_var', action='store', type=float, default=0.3,
        help='device variation [std] before write and verify')
parser.add_argument('--write_var', action='store', type=float, default=0.03,
        help='device variation [std] after write and verify')
parser.add_argument('--rate_zero', action='store', type=float, default=0.03,
        help='pepper rate, rate of noise being zero')
parser.add_argument('--rate_max', action='store', type=float, default=0.03,
        help='salt rate, rate of noise being one')
parser.add_argument('--noise_type', action='store', default="Gaussian",
        help='type of noise used')
parser.add_argument('--device', action='store', default="cuda:0",
        help='device used')
parser.add_argument('--verbose', action='store', type=str2bool, default=False,
        help='see training process')
parser.add_argument('--model', action='store', default="MLP4", choices=["MLP3", "MLP3_2", "MLP4", "LeNet", "CIFAR", "Res18", "TIN", "QLeNet", "QCIFAR", "QRes18", "QDENSE", "QTIN", "QVGG", "Adv", "QVGGIN", "QResIN"],
        help='model to use')
parser.add_argument('--header', action='store', default=None,
        help='use which saved state dict')
parser.add_argument('--div', action='store', type=int, default=1,
        help='division points for second')
parser.add_argument('--use_tqdm', action='store',type=str2bool, default=False,
        help='whether to use tqdm')

args = parser.parse_args()

def main():
    # print("{}".format(args).replace(', ', ',\n'))
    print(args)

    # if re.match('cuda:\d', args.device):
    #     cuda_num = args.device.split(':')[1]
    #     os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if you're using MBP M1, you can also use "mps"
    device = torch.device(args.device)
    print(device)

    # create related path
    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)

    args.poisoning_rate = 0
    data_loader_train_file, data_loader_val_clean_file, data_loader_val_poisoned_file = get_poision_datasets(args, args.batch_size, args.num_workers)
    data_loader_train = []
    for i in data_loader_train_file:
        data_loader_train.append(i)
    data_loader_val_clean = []
    for i in data_loader_val_clean_file:
        data_loader_val_clean.append(i)
    data_loader_val_poisoned = []
    for i in data_loader_val_poisoned_file:
        data_loader_val_poisoned.append(i)


    # model = BadNet(input_channels=1, output_num=args.nb_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optimizer_picker(args.optimizer, model.parameters(), lr=args.lr)
    model = get_model(args)
    model, optimizer, w_optimizer, scheduler = prepare_model(model, device, args)
    model_group = model, criterion, optimizer, scheduler, device, data_loader_train, data_loader_val_clean
    header = time.time()
    MTrain(model_group, args.epochs, header, args.noise_type, args.dev_var, args.rate_max, args.rate_zero, 0., True, N=8, m=1)
    state_dict = torch.load(f"tmp_best_{header}.pt")
    model.load_state_dict(state_dict)
    print(CEval(model_group))
    new_state_dict = model.state_dict()
    torch.save(new_state_dict, f"saved_B_{header}.pt")

if __name__ == "__main__":
    main()
