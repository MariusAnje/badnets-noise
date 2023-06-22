import torch
import torch.nn as nn
import torch.optim as optim
from models import modules
from tqdm import tqdm
import numpy as np
from models.modules import NConv2d, NLinear, SConv2d, SLinear, NModule, SModule
from models.qmodules import QSConv2d, QSLinear, QNConv2d, QNLinear
from deeplearning import evaluate_badnets

class BadAttack():
    def __init__(self, model, criterion, lr, steps, device, use_tqdm=False) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = torch.optim.SGD(self.get_bad(), lr=0.1)
        self.lr = lr
        self.device = device
        self._max = 0
        self.steps = steps
        self.use_tqdm = use_tqdm

    def get_bad(self):
        w = []
        for m in self.model.modules():
            if isinstance(m, modules.NModule) or isinstance(m, modules.SModule):
                m.bad.requires_grad_()
                w.append(m.bad)
        return w
    
    def collect_bad_tensor(self):
        flag = True
        for m in self.model.modules():
            if isinstance(m, modules.NModule) or isinstance(m, modules.SModule):
                if flag:
                    w = m.bad.view(-1)
                    flag = False
                else:
                    w = torch.cat([w, m.bad.view(-1)])
        return w
    
    def bad_l2(self):
        w = self.collect_bad_tensor()
        return torch.linalg.norm(w, 2) / np.sqrt(len(w))
    
    def bad_linf(self):
        w = self.collect_bad_tensor()
        return torch.linalg.norm(w, torch.inf)
    
    def bad_max(self):
        w = self.collect_bad_tensor()
        return w.abs().max()
    
    def collect_loss_ori(self, testloader):
        self.testloader = testloader
        if not isinstance(testloader.sampler, torch.utils.data.sampler.SequentialSampler):
            raise NotImplementedError("Don't use random sampler for dataloader")
        loss_set = []
        for images, labels in testloader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criteria(outputs, labels)
            loss_set.append(loss.item())
        self.loss_set = loss_set

    def cal_loss_l2(self, i, images, labels):
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self.model(images)
        loss = self.criteria(outputs, labels)
        return (loss - self.loss_set[i]).pow(2)
    
    def total_loss_l2(self):
        running_l2 = 0.0
        for i, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criteria(outputs, labels)
            l2 = (loss - self.loss_set[i]).pow(2)
            running_l2 += l2.item()
        return running_l2 / (i+1)
    
    def attack(self, data_loader_train, data_loader_val_clean, data_loader_val_poisoned):
        if self.use_tqdm:
            loader = tqdm(range(self.steps))
        else:
            loader = range(self.steps)
        for i in loader:
            self.attack_one_epoch(data_loader_train)
            test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, self.model, self.device)
            if self.use_tqdm:
                loader.set_description(f"Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}, Dist: {self.bad_max():.4f}")
            # print(f"#Epoch: [{i:03d}], Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}, Distance: {self.bad_max():.4f}")
        test_stats["dist"] = self.bad_max()
        return test_stats

class PGD(BadAttack):
    def attack_one_epoch(self, data_loader):
        running_loss = 0
        criterion, optimizer, device = self.criterion, self.optimizer, self.device
        self.model.train()
        # for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        for step, (batch_x, batch_y) in enumerate(data_loader):
            optimizer.zero_grad()
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            output = self.model(batch_x) # get predict label of batch_x
            loss = criterion(output, batch_y)
            loss.backward()
            self._max = 0
            running_loss += loss
            for m in self.model.modules():
                if isinstance(m, NModule) or isinstance(m, SModule):
                    m.bad.data -= m.op.weight.grad.data / m.op.weight.grad.data.abs().max() * self.lr
                    self._max = max(m.bad.data.max().item(), self._max)
        return {
                "loss": running_loss.item() / len(data_loader),
                }

class FGSM(BadAttack):
    def attack_one_epoch(self, data_loader):
        running_loss = 0
        criterion, optimizer, device = self.criterion, self.optimizer, self.device
        self.model.train()
        # for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        for step, (batch_x, batch_y) in enumerate(data_loader):
            optimizer.zero_grad()
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            output = self.model(batch_x) # get predict label of batch_x
            loss = criterion(output, batch_y)
            loss.backward()
            self._max = 0
            running_loss += loss
            for m in self.model.modules():
                if isinstance(m, NModule) or isinstance(m, SModule):
                    m.bad.data -= m.op.weight.grad.data.sign() * self.lr
                    self._max = max(m.bad.data.max().item(), self._max)
        return {
                "loss": running_loss.item() / len(data_loader),
                }

class LM(BadAttack):
    def __init__(self, model, criterion, lr, c, steps, device, use_tqdm=False) -> None:
        super().__init__(model, criterion, lr, steps, device, use_tqdm)
        self.c = c
        self.optimizer = torch.optim.Adam(self.get_bad(), lr=lr)

    def attack_one_epoch(self, data_loader):
        running_loss = 0
        criterion, optimizer, device = self.criterion, self.optimizer, self.device
        self.model.train()
        # for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            output = self.model(batch_x) # get predict label of batch_x
            loss = criterion(output, batch_y)
            cost = loss * self.c + self.bad_max()
            cost.backward()
            running_loss += loss
        return {
                "loss": running_loss.item() / len(data_loader),
                }
    
    def attack(self, data_loader_train, data_loader_val_clean, data_loader_val_poisoned):
        if self.use_tqdm:
            loader = tqdm(range(self.steps))
        else:
            loader = range(self.steps)
        for i in loader:
            self.optimizer.zero_grad()
            self.attack_one_epoch(data_loader_train)
            self.optimizer.step()
            test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, self.model, self.device)
            if self.use_tqdm:
                loader.set_description(f"Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}, Dist: {self.bad_max():.4f}")
            # print(f"#Epoch: [{i:03d}], Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}, Distance: {self.bad_max():.4f}")
        test_stats["dist"] = self.bad_max()
        return test_stats

def binary_search_dist(search_runs, dataloader, target_metric, attacker_class, model, criterion, init_c, steps, lr, device, verbose=True, use_tqdm=False):
    start_flag = True
    low = 0
    high = init_c
    mid = init_c
    final_accuracy = 0.0
    final_c = init_c
    final_max = None
    final_l2 = None
    for _ in range(search_runs):
        model.clear_noise()
        model.clear_bad()
        attacker = attacker_class(model, criterion, lr, mid, steps, device, use_tqdm)
        data_loader_train, data_loader_val_clean, data_loader_val_poisoned = dataloader
        test_stats = attacker.attack(data_loader_train, data_loader_val_clean, data_loader_val_poisoned)
        w = attacker.get_bad()
        this_max = attacker.bad_max().item()
        this_l2 = attacker.bad_l2().item()
        this_accuracy = test_stats["clean_acc"] + test_stats["asr"]
        clean = test_stats["clean_acc"]
        asr = test_stats["asr"]
        metric = attacker.bad_max().item()
        if verbose:
            print(f"C: {mid:.4e}, clean: {clean:.4f}, asr: {asr:.4f}, add: {this_accuracy:.4f}, l2: {this_l2:.4f},  max: {this_max:.4f}")
            pass
        if metric < target_metric:
            if start_flag:
                mid = mid * 10
                high = mid
            else:
                low = mid
                mid = (low + high) / 2
        else:
            final_c   = mid
            final_max = this_max
            final_l2  = this_l2
            final_accuracy = this_accuracy
            if low == 0:
                mid = mid / 10
                if not start_flag:
                    high = high / 10
            else:
                high = mid
                mid = (low + high) / 2
            start_flag = False
            if np.abs(metric-target_metric) < 1e-5:
                break
    return final_accuracy, final_max, final_l2, final_c

