import numpy as np
import argparse
import os

class FileData():
    def __init__(self, result_form, args):
        self.parameters = {}
        self.results = result_form(args)

class FolderData():
    def __init__(self):
        self.parameter_names = []

class AttackAccuracyData():
    def __init__(self):
        self.distance = 0
        self.acc_free = {}
        self.acc_attack = {}

class AttackAccuracyDataFile():
    def __init__(self, args):
        self.results = {}
        self.args = args

    def add(self, data:AttackAccuracyData):
        self.results[data.distance] = data
    
    def best_result_metric(self, acc_free, acc_attack, n_classes=10):
        # return self.args.A * acc_free["clean_acc"] - self.args.B * acc_free["asr"] + self.args.A * acc_attack["clean_acc"] + self.args.C * acc_attack["asr"]
        return self.args.A * acc_attack["clean_acc"] + self.args.C * acc_attack["asr"]
    
    def fetch_best(self, distance):
        best_metric = -10000000
        best_key = None
        for key in self.results:
            if key < distance:
                acc_free = self.results[key].acc_free
                acc_attack = self.results[key].acc_attack
                metric = self.best_result_metric(acc_free, acc_attack)
                if metric > best_metric:
                    best_metric = metric
                    best_key = key
        if best_key is None:
            lol = []
            for key in self.results:
                lol.append(key)
            best_key = min(lol)
        return self.results[best_key]
    
    def fetch(self, distance):
        def find_closest(ln, n):
            closest_num = min(ln, key=lambda x: abs(x - n))
            return closest_num
        the_key = find_closest(self.results.keys(), distance)
        return self.results[the_key]

class GroupedData():
    def __init__(self, key_params):
        self.key_params = key_params
        self.grouped = {}

    def dict_to_string(self, param_dict, key_params):
        new_dict = {}
        for param in key_params:
            new_dict[param] = param_dict[param]
        return str(new_dict)

    def add(self, data):
        index = self.dict_to_string(data.parameters, self.key_params)
        if index in self.grouped:
            self.grouped[index].append(data)
        else:
            self.grouped[index] = [data]

    def fetch_average(self, distance):
        name_list = []
        param_data = {}
        for param in self.grouped:
            data_list = []
            for file in self.grouped[param]:
                this_list = []
                this_data = file.results.fetch_best(distance)
                acc_free = this_data.acc_free
                acc_attack = this_data.acc_attack
                if len(name_list) == 0:
                    for key in acc_free:
                        name_list.append("free " + key)
                    for key in acc_attack:
                        name_list.append("attack " + key)
                for key in acc_free:
                    this_list.append(acc_free[key])
                for key in acc_attack:
                    this_list.append(acc_attack[key])
                data_list.append(this_list)
            data_list = np.array(data_list).mean(axis=0)
            param_data[param] = data_list
        return param_data, name_list

def parse_namespace(line:str):
    namespace = {}
    if "Namespace(" in line:
        line = line[10:-1]
        all_names = line.split(",")
        for item in all_names:
            name, data = item.split("=")
            name = name.replace(" ", "")
            namespace[name] = data
    return namespace

def parse_acc_one_line(line:str, indicator:str):
    new_line = line[line.find(indicator)+len(indicator):]
    acc_data = new_line.split(",")[0].split("/")
    clean_acc, asr = float(acc_data[0]), float(acc_data[1])
    res = {"clean_acc": clean_acc, "asr": asr}
    return res

def read_using_keyword(line:str, keyword:str, offset:int):
    index = line.find(keyword) + len(keyword)
    return line[index:index+offset]

def read_file(file_name:str, args):
    data_storage = FileData(AttackAccuracyDataFile, args)
    with open(file_name) as f:
        file_data = f.read()
    for line in file_data.splitlines():
        if "Namespace" in line:
            data_storage.parameters = parse_namespace(line)
        elif "epoch:" in line:
            this_epoch_data = AttackAccuracyData()
            indicator_free = "ori acc/asr:"
            indicator_atk  = "clean acc/asr:"
            this_epoch_data.acc_free = parse_acc_one_line(line, indicator_free)
            this_epoch_data.acc_attack = parse_acc_one_line(line, indicator_atk)
            this_epoch_data.distance = float(read_using_keyword(line, keyword="dist: ", offset=6))
            data_storage.results.add(this_epoch_data)
    return data_storage
# data_storage = read_file("CW.o868884.3")

def group_same_settings(folder_data:list):
    key_params = ["attack_c", "attack_lr", "attack_w_lr"]
    backups = ["attack_runs", "attack_method", "attack_start"]
    param_list = folder_data[0].parameters.keys()
    grouped_data = GroupedData(key_params)
    for data in folder_data:
        grouped_data.add(data)
    return grouped_data

def get_folder(folder_dir:str, args):
    folder_data = []
    files = os.listdir(folder_dir)
    for fn in files:
        if "CW" in fn:
            abs_fn = os.path.join(folder_dir, fn)
            folder_data.append(read_file(abs_fn, args))
    return folder_data

def best_result_metric(data, args):
    # return args.A * data[0] - args.B * data[1] + args.A * data[2] + args.C * data[3]
    return args.A * data[2] + args.C * data[3]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='./')
    parser.add_argument('--A', action='store',type=float, default=1)
    parser.add_argument('--B', action='store',type=float, default=1)
    parser.add_argument('--C', action='store',type=float, default=1)
    parser.add_argument('--dist', action='store',type=float, default=0.03)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    distances = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]
    lol1 = ""
    lol2 = ""
    for dist in distances:
        folder_data = get_folder(args.dir, args)
        grouped_data = group_same_settings(folder_data)
        res_dict, name_list = grouped_data.fetch_average(dist)

        best_result = -1000000
        for key in res_dict:
            data = res_dict[key]
            mt = best_result_metric(data, args)
            if mt > best_result:
                best_result = mt
                best_key = key
            # print(key)
            # print(res_dict[key])

        # print(best_key)
        # print(res_dict[best_key])
        print(f"Dist: {dist:.3f}, acc: {res_dict[best_key][2]:.4f}, asr: {res_dict[best_key][3]:.4f}")
        lol1 += f"{res_dict[best_key][2]:.4f}\n"
        lol2 += f"{res_dict[best_key][3]:.4f}\n"
    print(lol1, lol2)

if __name__ == "__main__":
    main()