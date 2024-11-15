import torch
import time
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch import nn
from models.modules import  SModule, NModule
from models import modules
from models.models import SCrossEntropyLoss, SMLP3, SMLP4, SLeNet, CIFAR, FakeSCrossEntropyLoss, SAdvNet
from models.qmodels import QSLeNet, QCIFAR
from models import resnet
from models import qresnet
from models import qvgg
from models import qdensnet
from models import qresnetIN
from torch import optim
import logging
from dataset import build_poisoned_training_set, build_testset
from deeplearning import evaluate_badnets

def backup_BN(model):
    backup_list = {}
    state_dict = model.state_dict()
    for key in state_dict:
        if ".bn" in key:
            backup_list[key] = torch.zeros_like(state_dict[key].data)
            backup_list[key].data += state_dict[key].data
    return backup_list

def load_BN(model, backup_list):
    state_dict = model.state_dict()
    for key in backup_list:
        state_dict[key].data = torch.zeros_like(backup_list[key].data)
        state_dict[key].data += backup_list[key].data
    model.load_state_dict(state_dict)

def switch_BN(model, backup_list):
    new_backup = backup_BN(model)
    load_BN(model, backup_list)
    return new_backup

def get_memory_dataset(file_set_list:list):
    newset_list = []
    for file_set in file_set_list:
        newset = []
        for i in file_set:
            newset.append(i)
        newset_list.append(newset)
    return newset_list

def get_poision_datasets(args, BS, NW):
    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)
    dataset_train, dataset_val_clean, dataset_val_poisoned = get_memory_dataset([dataset_train, dataset_val_clean, dataset_val_poisoned])
    data_loader_train        = torch.utils.data.DataLoader(dataset_train,         batch_size=BS, shuffle=True, num_workers=NW)
    data_loader_val_clean    = torch.utils.data.DataLoader(dataset_val_clean,     batch_size=BS, shuffle=False, num_workers=NW)
    data_loader_val_poisoned = torch.utils.data.DataLoader(dataset_val_poisoned,  batch_size=BS, shuffle=False, num_workers=NW)
    return data_loader_train, data_loader_val_clean, data_loader_val_poisoned

def get_dataset(args, BS, NW):
    if args.model == "CIFAR" or args.model == "Res18" or args.model == "QCIFAR" or args.model == "QRes18" or args.model == "QDENSE":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        transform = transforms.Compose(
        [transforms.ToTensor(),
        #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            normalize])
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
                ])
        trainset = torchvision.datasets.CIFAR10(root='~/Private/data', train=True, download=False, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True, num_workers=4)
        secondloader = torch.utils.data.DataLoader(trainset, batch_size=BS//args.div, shuffle=False, num_workers=4)
        testset = torchvision.datasets.CIFAR10(root='~/Private/data', train=False, download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS, shuffle=False, num_workers=4)
    elif args.model == "TIN" or args.model == "QTIN" or args.model == "QVGG":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
                [transforms.ToTensor(),
                 normalize,
                ])
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, 4),
                transforms.ToTensor(),
                normalize,
                ])
        args.data_path = ['~/Private/data/tiny-imagenet-200/train',
                          '~/Private/data/tiny-imagenet-200/val',]
        trainset = torchvision.datasets.ImageFolder(root='~/Private/data/tiny-imagenet-200/train', transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True, num_workers=4)
        secondloader = torch.utils.data.DataLoader(trainset, batch_size=BS//args.div, shuffle=False, num_workers=4)
        testset = torchvision.datasets.ImageFolder(root='~/Private/data/tiny-imagenet-200/val',  transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS, shuffle=False, num_workers=4)
    elif args.model == "QVGGIN" or args.model == "QResIN":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        pre_process = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        pre_process += [
            transforms.ToTensor(),
            normalize
        ]

        trainset = torchvision.datasets.ImageFolder('/data/data/share/imagenet/train',
                                transform=transforms.Compose(pre_process))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.ImageFolder('/data/data/share/imagenet/val',
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize
                                ]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                                    shuffle=False, num_workers=4)
    else:
        trainset = torchvision.datasets.MNIST(root='~/Private/data', train=True,
                                                download=False, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                                shuffle=True, num_workers=NW)
        secondloader = torch.utils.data.DataLoader(trainset, batch_size=BS//args.div,
                                                shuffle=False, num_workers=NW)

        testset = torchvision.datasets.MNIST(root='~/Private/data', train=False,
                                            download=False, transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                                    shuffle=False, num_workers=NW)
    return trainloader, secondloader, testloader

def get_model(args):
    if args.model == "MLP3":
        model = SMLP3()
    elif args.model == "MLP3_2":
        model = SMLP3()
    elif args.model == "MLP4":
        model = SMLP4()
    elif args.model == "LeNet":
        model = SLeNet()
    elif args.model == "CIFAR":
        model = CIFAR()
    elif args.model == "Res18":
        model = resnet.resnet18(num_classes = 10)
    elif args.model == "TIN":
        model = resnet.resnet18(num_classes = 200)
    elif args.model == "QLeNet":
        model = QSLeNet()
    elif args.model == "QCIFAR":
        model = QCIFAR()
    elif args.model == "QRes18":
        model = qresnet.resnet18(num_classes = 10)
    elif args.model == "QDENSE":
        model = qdensnet.densenet121(num_classes = 10)
    elif args.model == "QTIN":
        model = qresnet.resnet18(num_classes = 200)
    elif args.model == "QVGG":
        model = qvgg.vgg16(num_classes = 1000)
    elif args.model == "Adv":
        model = SAdvNet()
    elif args.model == "QVGGIN":
        model = qvgg.vgg16(num_classes = 1000)
    elif args.model == "QResIN":
        model = qresnetIN.resnet18(num_classes = 1000)
    else:
        NotImplementedError
    return model

def prepare_model(model, device, args):
    model.to(device)
    for m in model.modules():
        if isinstance(m, modules.FixedDropout) or isinstance(m, modules.NFixedDropout) or isinstance(m, modules.SFixedDropout):
            m.device = device
    model.push_S_device()
    model.clear_noise()
    model.clear_bad()
    model.clear_mask()
    model.to_first_only()
    model.de_select_drop()
    if "TIN" in args.model or "Res" in args.model or "VGG" in args.model or "DENSE" in args.model:
    # if "TIN" in args.model or "Res" in args.model:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [1000])
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60])
    warm_optimizer = optim.SGD(model.parameters(), lr=1e-3)
    return model, optimizer, warm_optimizer, scheduler

def copy_model(old_model, args):
    new_model = get_model(args)
    state_dict = old_model.state_dict()
    for key in state_dict.keys():
        if "weight" in key:
            device = state_dict[key].device
            break
    new_model, optimizer, warm_optimizer, scheduler = prepare_model(new_model, device, args)
    new_model.load_state_dict(old_model.state_dict())
    return new_model, optimizer, warm_optimizer, scheduler

def get_logger(filepath=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)
    if filepath is not None:
        file_handler = logging.FileHandler(filepath+'.log', mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)
    return logger




def UpdateBN(model_group):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.train()
    total = 0
    correct = 0
    # model.clear_noise()
    with torch.no_grad():
        # for images, labels in tqdm(testloader, leave=False):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)

def TUpdateBN(t_model_group):
    t_model, criteriaF, t_optimizer, w_optimizer, t_scheduler, device, trainloader, testloader = t_model_group
    for i in range(len(t_model)):
        model = t_model[i]
        model.train()
        total = 0
        correct = 0
        model.clear_noise()
        with torch.no_grad():
            for images, labels in trainloader:
                model.clear_noise()
                # model.set_SPU(s_rate, p_rate, dev_var)
                images, labels = images.to(device), labels.to(device)
                # images = images.view(-1, 784)
                outputs = model(images)

def CEval(model_group):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.eval()
    total = 0
    correct = 0
    # model.clear_noise()
    with torch.no_grad():
        # for images, labels in tqdm(testloader, leave=False):
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().item()

def CEval_Dist(model_group, num_classes=10):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.eval()
    total = 0
    correct = 0
    res_dist = torch.LongTensor([0 for _ in range(num_classes)])
    crr_dist = torch.LongTensor([0 for _ in range(num_classes)])
    # model.clear_noise()
    with torch.no_grad():
        # for images, labels in tqdm(testloader, leave=False):
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            predict_list = predictions.tolist()
            for i in range(len(res_dist)):
                res_dist[i] += predict_list.count(i)
            for i in range(len(predict_list)):
                if correction[i] == True:
                    crr_dist[predict_list[i]] += 1
            correct += correction.sum()
            total += len(correction)
    print(f"Correction dict: {crr_dist.tolist()}")
    return (correct/total).cpu().item(), res_dist

def NEval(model_group, dev_var, write_var):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.eval()
    total = 0
    correct = 0
    model.clear_noise()
    with torch.no_grad():
        model.set_noise(dev_var, write_var)
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().item()

def NEachEval(model_group, dev_var, write_var):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.eval()
    total = 0
    correct = 0
    model.clear_noise()
    with torch.no_grad():
        for images, labels in testloader:
            model.clear_noise()
            model.set_noise(dev_var, write_var)
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().item()

def MEval(model_group, noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.eval()
    total = 0
    correct = 0
    model.clear_noise()
    with torch.no_grad():
        model.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().item()

def MBEval(model_group, noise_type, dev_var, rate_max, rate_zero, write_var, acc_th=0.5, **kwargs):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.eval()
    total = 0
    correct = 0
    acc_list = []
    with torch.no_grad():
        for images, labels in testloader:
            model.clear_noise()
            model.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
            batch_acc = (correction.sum() / len(correction)).item()
            if batch_acc < acc_th:
                acc_list.append(CEval(model_group))
            else:
                acc_list.append(batch_acc)
    return acc_list

def MEachEval(model_group, noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.eval()
    total = 0
    correct = 0
    model.clear_noise()
    with torch.no_grad():
        for images, labels in testloader:
            model.clear_noise()
            model.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
            # model.set_SPU(s_rate, p_rate, dev_var)
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().item()

def TMEval(t_model_group, noise_type, dev_var_list, rate_max, rate_zero, write_var, **kwargs):
    t_model, criteriaF, t_optimizer, w_optimizer, t_scheduler, device, trainloader, testloader = t_model_group
    acc_list = []
    for i in range(len(t_model)):
        model = t_model[i]
        model.eval()
        total = 0
        correct = 0
        model.clear_noise()
        with torch.no_grad():
            model.set_noise_multiple(noise_type, dev_var_list[i], rate_max, rate_zero, write_var, **kwargs)
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if len(outputs) == 2:
                    outputs = outputs[0]
                predictions = outputs.argmax(dim=1)
                correction = predictions == labels
                correct += correction.sum()
                total += len(correction)
            acc_list.append((correct/total).cpu().item())
    return acc_list

def TMEachEval(t_model_group, noise_type, dev_var_list, rate_max, rate_zero, write_var, **kwargs):
    t_model, criteriaF, t_optimizer, w_optimizer, t_scheduler, device, trainloader, testloader = t_model_group
    acc_list = []
    for i in range(len(t_model)):
        model = t_model[i]
        model.eval()
        total = 0
        correct = 0
        model.clear_noise()
        with torch.no_grad():
            for images, labels in testloader:
                model.clear_noise()
                model.set_noise_multiple(noise_type, dev_var_list[i], rate_max, rate_zero, write_var, **kwargs)
                # model.set_SPU(s_rate, p_rate, dev_var)
                images, labels = images.to(device), labels.to(device)
                # images = images.view(-1, 784)
                outputs = model(images)
                if len(outputs) == 2:
                    outputs = outputs[0]
                predictions = outputs.argmax(dim=1)
                correction = predictions == labels
                correct += correction.sum()
                total += len(correction)
            acc_list.append((correct/total).cpu().item())
    return acc_list

def TCEval(t_model_group):
    t_model, criteriaF, optimizer, w_optimizer, scheduler, device, trainloader, testloader = t_model_group
    acc_list = []
    for model in t_model:
        model.eval()
        total = 0
        correct = 0
        # model.clear_noise()
        with torch.no_grad():
            # for images, labels in tqdm(testloader, leave=False):
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                # images = images.view(-1, 784)
                outputs = model(images)
                if len(outputs) == 2:
                    outputs = outputs[0]
                predictions = outputs.argmax(dim=1)
                correction = predictions == labels
                correct += correction.sum()
                total += len(correction)
            acc_list.append((correct/total).cpu().numpy())
    return acc_list

def NTrain(model_group, epochs, header, dev_var=0.0, write_var=0.0, verbose=False):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    best_acc = 0.0
    for i in range(epochs):
        model.train()
        running_loss = 0.
        # for images, labels in tqdm(trainloader):
        for images, labels in trainloader:
            model.clear_noise()
            model.set_noise(dev_var, write_var)
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            loss = criteriaF(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        test_acc = NEachEval(model_group, dev_var, write_var)
        # test_acc = CEval()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"tmp_best_{header}.pt")
        if verbose:
            print(f"epoch: {i:-3d}, test acc: {test_acc:.4f}, loss: {running_loss / len(trainloader):.4f}")
        scheduler.step()

def MTrain(model_group, epochs, header, noise_type, dev_var, rate_max, rate_zero, write_var, verbose=False, **kwargs):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    best_acc = 0.0
    set_noise = True
    for i in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.
        # for images, labels in tqdm(trainloader):
        for images, labels in trainloader:
            model.clear_noise()
            if set_noise:
                model.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criteriaF(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        test_acc = MEachEval(model_group, noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
        model.clear_noise()
        noise_free_acc = CEval(model_group)
        # if noise_free_acc - test_acc < -0.02:
        #     set_noise = False
        # else:
        #     set_noise = True
        # if noise_free_acc - test_acc < -0.02:
        #     UpdateBN(model_group)
        #     noise_free_acc = CEval(model_group)
        if set_noise:
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f"tmp_best_{header}.pt")
        if verbose:
            end_time = time.time()
            print(f"epoch: {i:-3d}, test acc: {test_acc:.4f}, clean acc: {noise_free_acc:.4f}, noise set: {set_noise}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")
        scheduler.step()

def AMTrain(model_group, attacker, data_loader_train_poisoned, data_loader_val_poisoned, epochs, atk_start, header, noise_type, dev_var, rate_max, rate_zero, write_var, verbose=False, **kwargs):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    best_acc = 0.0
    set_noise = True
    for i in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.
        model.set_mask_zero()
        for images, labels in trainloader:
            model.clear_noise()
            if set_noise:
                model.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criteriaF(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        test_acc = MEachEval(model_group, noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
        model.clear_noise()
        model.clear_mask()
        if i >= atk_start:
            test_stats = attacker.attack(data_loader_train_poisoned, testloader, data_loader_val_poisoned)
            clean = test_stats["clean_acc"]
            asr = test_stats["asr"]
            dist = test_stats["dist"]
        model.set_mask_zero()
        no_atk_test_stats = evaluate_badnets(testloader, data_loader_val_poisoned, model, device)
        no_atk_clean, no_atk_asr = no_atk_test_stats["clean_acc"], no_atk_test_stats["asr"]
        model.clear_mask()
        if i < atk_start:
            clean = no_atk_clean
            asr = no_atk_asr
            dist = 0
        test_acc = clean + asr
        if set_noise:
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f"tmp_best_{header}.pt")
        if verbose:
            end_time = time.time()
            print(f"epoch: {i:-3d}, test acc: {test_acc:.4f}, ori acc/asr: {no_atk_clean:.4f}/{no_atk_asr:.4f}, clean acc/asr: {clean:.4f}/{asr:.4f}, dist: {dist:.4f}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")
        scheduler.step()

def AMTrainBN(model_group, attacker, data_loader_train_poisoned, data_loader_val_poisoned, epochs, atk_start, header, noise_type, dev_var, rate_max, rate_zero, write_var, verbose=False, **kwargs):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    best_acc = 0.0
    set_noise = True
    for i in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.
        model.set_mask_zero()
        for images, labels in trainloader:
            model.clear_noise()
            if set_noise:
                model.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criteriaF(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        test_acc = MEachEval(model_group, noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
        model.clear_noise()
        model.clear_mask()
        if i >= atk_start:
            if i == atk_start:
                # clean_BN = backup_BN(model)
                atk_BN = backup_BN(model)
            clean_BN = switch_BN(model, atk_BN)
            test_stats = attacker.attack(data_loader_train_poisoned, testloader, data_loader_val_poisoned)
            clean = test_stats["clean_acc"]
            asr = test_stats["asr"]
            dist = test_stats["dist"]
            atk_BN = switch_BN(model, clean_BN)
            # clean_BN = switch_BN(model, atk_BN)
        model.set_mask_zero()
        no_atk_test_stats = evaluate_badnets(testloader, data_loader_val_poisoned, model, device)
        no_atk_clean, no_atk_asr = no_atk_test_stats["clean_acc"], no_atk_test_stats["asr"]
        model.clear_mask()
        if i < atk_start:
            clean = no_atk_clean
            asr = no_atk_asr
            dist = 0
        test_acc = clean + asr
        if set_noise:
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f"tmp_best_{header}.pt")
        if verbose:
            end_time = time.time()
            print(f"epoch: {i:-3d}, test acc: {test_acc:.4f}, ori acc/asr: {no_atk_clean:.4f}/{no_atk_asr:.4f}, clean acc/asr: {clean:.4f}/{asr:.4f}, dist: {dist:.4f}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")
        scheduler.step()
        if "dist_th" in kwargs:
            if dist > kwargs["dist_th"]:
                break

def DMTrain(model_group, epochs, header, noise_type, dev_var, rate_max, rate_zero, write_var, verbose=False, **kwargs):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    best_acc = 0.0
    set_noise = True
    for i in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.
        # for images, labels in tqdm(trainloader):
        for images, labels in trainloader:
            model.clear_noise()
            if set_noise:
                model.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criteriaF(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        test_acc = MEachEval(model_group, noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
        model.clear_noise()
        noise_free_acc = CEval(model_group)

        if set_noise:
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f"tmp_best_{header}.pt")
        if verbose:
            end_time = time.time()
            cross, act = criteriaF.summary()
            print(f"epoch: {i:-3d}, test acc: {test_acc:.4f}, clean acc: {noise_free_acc:.4f}, noise set: {set_noise}, loss: {cross:.4f}, sense: {act:.4f}, used time: {end_time - start_time:.4f}")
        scheduler.step()

def HMTrain(model_group, epochs, header, noise_type, dev_var, rate_max, rate_zero, write_var, 
            eval_noise_type, eval_dev_var, eval_rate_max, eval_rate_zero, verbose=False, **kwargs):
    
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    best_acc = 0.0
    for i in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.
        # for images, labels in tqdm(trainloader):
        for images, labels in trainloader:
            model.clear_noise()
            model.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criteriaF(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        test_acc = MEachEval(model_group, eval_noise_type, eval_dev_var, eval_rate_max, eval_rate_zero, write_var, **kwargs)
        model.clear_noise()
        noise_free_acc = CEval(model_group)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"tmp_best_{header}.pt")
        if verbose:
            end_time = time.time()
            print(f"epoch: {i:-3d}, test acc: {test_acc:.4f}, clean acc: {noise_free_acc:.4f}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")
        scheduler.step()

def TNMTrain(three_model_group, warm_epochs, epochs, header, noise_type, dev_var_start, dev_var_end, rate_max, rate_zero, write_var, attack_runs, attack_dist, logger=None, verbose=False, **kwargs):
    t_model, criteriaF, t_optimizer, w_optimizer, t_scheduler, device, trainloader, testloader = three_model_group
    best_acc = 0.0
    start, end = dev_var_start, dev_var_end
    merged_flag = False

    for ep in range(warm_epochs + epochs):
        if end - start < 1e-4 and not merged_flag:
            three_model_group = [t_model[1]], criteriaF, [t_optimizer[1]], [w_optimizer[1]], [t_scheduler[1]], device, trainloader, testloader
            t_model, criteriaF, t_optimizer, w_optimizer, t_scheduler, device, trainloader, testloader = three_model_group
            merged_flag = True

        mid = (start + end)/2
        oForth = (end - start) / 4
        left = start + oForth
        right = end - oForth
        dev_var_list = [left, mid, right]
        start_time = time.time()
        for i in range(len(t_model)):
            t_model[i].train()
        running_loss = 0.
        # for images, labels in tqdm(trainloader):
        for images, labels in trainloader:
            for i in range(len(t_model)):
                t_model[i].clear_noise()
                t_model[i].set_noise_multiple(noise_type, dev_var_list[i], rate_max, rate_zero, write_var, **kwargs)
                t_optimizer[i].zero_grad()
                images, labels = images.to(device), labels.to(device)
                outputs = t_model[i](images)
                loss = criteriaF(outputs,labels)
                loss.backward()
                if ep >= warm_epochs:
                    t_optimizer[i].step()
                else:
                    w_optimizer[i].step()
                running_loss += loss.item()
        test_acc = TMEachEval(three_model_group, noise_type, dev_var_list, rate_max, rate_zero, write_var, **kwargs)
        best_pgd_index = min(len(t_model) - 1, 1)
        if ep >= warm_epochs:
            TUpdateBN(three_model_group)
            pgd_acc = TMEachEval(three_model_group, "Gaussian", [attack_dist]*len(dev_var_list), rate_max, rate_zero, write_var, **kwargs)
            best_pgd_index = np.argmax(pgd_acc)

            if verbose:
                end_time = time.time()
                if logger is None:
                    print(f"epoch: {ep:-3d}, test acc: {test_acc[best_pgd_index]:.4f}, noise acc: {pgd_acc[best_pgd_index]:.4f}, start: {start:.4f}, end: {end:.4f}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")
                else:
                    logger.info(f"epoch: {ep:-3d}, test acc: {test_acc[best_pgd_index]:.4f}, noise acc: {pgd_acc[best_pgd_index]:.4f}, start: {start:.4f}, end: {end:.4f}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")
            
            if best_pgd_index == 0:
                end = right
            elif best_pgd_index == 1:
                start = left
                end = right
            else:
                start = left
            for i in range(len(t_model)):
                t_model[i].load_state_dict(t_model[best_pgd_index].state_dict())

            # test_acc = CEval()
            if pgd_acc[best_pgd_index] > best_acc:
                best_acc = pgd_acc[best_pgd_index]
                torch.save(t_model[best_pgd_index].state_dict(), f"tmp_best_{header}.pt")
            
            for i in range(len(t_scheduler)):
                t_scheduler[i].step()
        else:
            if verbose:
                end_time = time.time()
                if logger is None:
                    print(f"warm up epoch: {ep:-3d}, test acc: {test_acc[best_pgd_index]:.4f}, mid: {mid:.4f}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")
                else:
                    logger.info(f"warm up epoch: {ep:-3d}, test acc: {test_acc[best_pgd_index]:.4f}, mid: {mid:.4f}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")

def TQMTrain(three_model_group, warm_epochs, epochs, noise_epochs, quantile, header, 
             noise_type, dev_var_start, dev_var_end, train_max, train_zero, write_var, 
             attack_runs, test_noise_type, test_max, test_zero, attack_dist, logger=None, verbose=False, **kwargs):
    t_model, criteriaF, t_optimizer, w_optimizer, t_scheduler, device, trainloader, testloader, testloader_large = three_model_group
    three_model_group = t_model, criteriaF, t_optimizer, w_optimizer, t_scheduler, device, trainloader, testloader
    three_model_group_large = t_model, criteriaF, t_optimizer, w_optimizer, t_scheduler, device, trainloader, testloader_large
    best_acc = 0.0
    start, end = dev_var_start, dev_var_end
    merged_flag = False

    for ep in range(warm_epochs + epochs):
        if end - start < 1e-4 and not merged_flag:
            three_model_group = [t_model[1]], criteriaF, [t_optimizer[1]], [w_optimizer[1]], [t_scheduler[1]], device, trainloader, testloader
            three_model_group_large = [t_model[1]], criteriaF, [t_optimizer[1]], [w_optimizer[1]], [t_scheduler[1]], device, trainloader, testloader_large
            t_model, criteriaF, t_optimizer, w_optimizer, t_scheduler, device, trainloader, testloader = three_model_group
            merged_flag = True

        mid = (start + end)/2
        oForth = (end - start) / 4
        left = start + oForth
        right = end - oForth
        dev_var_list = [left, mid, right]
        start_time = time.time()
        for i in range(len(t_model)):
            t_model[i].train()
        running_loss = 0.
        # for images, labels in tqdm(trainloader):
        for images, labels in trainloader:
            for i in range(len(t_model)):
                t_model[i].clear_noise()
                t_model[i].set_noise_multiple(noise_type, dev_var_list[i], train_max, train_zero, write_var, **kwargs)
                t_optimizer[i].zero_grad()
                images, labels = images.to(device), labels.to(device)
                outputs = t_model[i](images)
                loss = criteriaF(outputs,labels)
                loss.backward()
                if ep >= warm_epochs:
                    t_optimizer[i].step()
                else:
                    w_optimizer[i].step()
                running_loss += loss.item()
        test_acc = TMEachEval(three_model_group, noise_type, dev_var_list, test_max, train_max, train_zero, **kwargs)
        best_pgd_index = min(len(t_model) - 1, 1)
        if ep >= warm_epochs:
            TUpdateBN(three_model_group)
            acc_list = []
            for _ in range(noise_epochs):
                tmp_acc = TMEval(three_model_group_large, test_noise_type, [attack_dist]*len(dev_var_list), test_max, test_zero, write_var, **kwargs)
                acc_list.append(tmp_acc)
            pgd_acc = np.quantile(np.array(acc_list), quantile, axis=0)
            best_pgd_index = np.argmax(pgd_acc)

            if verbose:
                end_time = time.time()
                if logger is None:
                    print(f"epoch: {ep:-3d}, test acc: {test_acc[best_pgd_index]:.4f}, noise acc: {pgd_acc[best_pgd_index]:.4f}, start: {start:.4f}, end: {end:.4f}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")
                else:
                    logger.info(f"epoch: {ep:-3d}, test acc: {test_acc[best_pgd_index]:.4f}, noise acc: {pgd_acc[best_pgd_index]:.4f}, start: {start:.4f}, end: {end:.4f}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")
            
            if best_pgd_index == 0:
                end = right
            elif best_pgd_index == 1:
                start = left
                end = right
            else:
                start = left
            for i in range(len(t_model)):
                t_model[i].load_state_dict(t_model[best_pgd_index].state_dict())

            # test_acc = CEval()
            if pgd_acc[best_pgd_index] > best_acc:
                best_acc = pgd_acc[best_pgd_index]
                torch.save(t_model[best_pgd_index].state_dict(), f"tmp_best_{header}.pt")
            
            for i in range(len(t_scheduler)):
                t_scheduler[i].step()
        else:
            if verbose:
                end_time = time.time()
                if logger is None:
                    print(f"warm up epoch: {ep:-3d}, test acc: {test_acc[best_pgd_index]:.4f}, mid: {mid:.4f}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")
                else:
                    logger.info(f"warm up epoch: {ep:-3d}, test acc: {test_acc[best_pgd_index]:.4f}, mid: {mid:.4f}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")

def str2bool(a):
    if a == "True":
        return True
    elif a == "False":
        return False
    else:
        raise NotImplementedError(f"{a}")

def f_act(targeted, outputs, labels, kappa=0, gamma=1e10):
    one_hot_labels = torch.eye(len(outputs[0]))[labels.cpu()].to(outputs.device)

    i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the second largest logit
    j = outputs[one_hot_labels.bool().cpu()]

    if targeted:
        # return -torch.clamp((i-j), min=-kappa).sum()
        return -torch.clamp((i-j), min=-kappa, max=gamma).sum()
    else:
        # return -torch.clamp((j-i), min=-kappa).sum()
        return -torch.clamp((j-i), min=-kappa, max=gamma).sum()


class LossCrossAct(nn.Module):
    def __init__(self, alpha, targeted=False, kappa=0, gamma=1e10) -> None:
        super().__init__()
        self._alpha = alpha
        self._targeted = targeted
        self._kappa = kappa
        self._gamma = gamma
        self._cross = nn.CrossEntropyLoss()
        self._act = f_act
        self.record = {"cross":[], "act":[]}
    
    def forward(self, outputs, labels):
        cr = self._cross(outputs, labels)
        act = self._act(self._targeted, outputs, labels, self._kappa, self._gamma)
        self.record["cross"].append(cr.item())
        self.record["act"].append(act.item())
        return cr - self._alpha * act

    def get_average(self):
        return np.mean(self.record["cross"]), np.mean(self.record["act"])
    
    def clear(self) -> None:
        self.record["cross"] = []
        self.record["act"] = []
    
    def summary(self):
        cross, act = self.get_average()
        self.clear()
        return cross, act

def get_bad(self):
    w = []
    for m in self.modules():
        if isinstance(m, modules.NModule) or isinstance(m, modules.SModule):
            m.bad.requires_grad_()
            w.append(m.bad)
    return w
