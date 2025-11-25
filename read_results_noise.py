import numpy as np
import argparse
import os

class DataEntry():
    def __init__(self) -> None:
        self.acc_b = []
        self.asr_b = []
        self.acc_p = []
        self.asr_p = []
    
    def add(self, acc_b, asr_b, acc_p, asr_p):
        self.acc_b.append(acc_b)
        self.asr_b.append(asr_b)
        self.acc_p.append(acc_p)
        self.asr_p.append(asr_p)

    def fetch(self):
        return np.mean(self.acc_b), np.mean(self.asr_b), np.mean(self.acc_p), np.mean(self.asr_p)

def read_using_keyword(line:str, keyword:str, offset:int):
    index = line.find(keyword) + len(keyword)
    return line[index:index+offset]

def read_file_data(dir, keyword="Dist: "):
    my_data = {}
    for j in range(2,3):
        fn = f"{dir}.{j}"
        with open(fn) as f:
            file_data = f.read().splitlines()
        for i, line in enumerate(file_data):
            if "Dist: " in line:
                noise_std = read_using_keyword(line, "Dist: ", 5)
                daddy = read_using_keyword(line, "ori acc/asr: ", 13)
                acc_b = float(daddy[:6])
                asr_b = float(daddy[7:])
                daddy = read_using_keyword(line, "noise acc/asr: ", 13)
                acc_p = float(daddy[:6])
                asr_p = float(daddy[7:])
                if not (noise_std in my_data):
                    my_data[noise_std] = (DataEntry(), DataEntry())
                my_data[noise_std][0].add(acc_b, asr_b, acc_p, asr_p)
                line2 = file_data[i+1]
                daddy = read_using_keyword(line2, "ori acc/asr: ", 13)
                acc_b = float(daddy[:6])
                asr_b = float(daddy[7:])
                daddy = read_using_keyword(line2, "noise acc/asr: ", 13)
                acc_p = float(daddy[:6])
                asr_p = float(daddy[7:])
                my_data[noise_std][1].add(acc_b, asr_b, acc_p, asr_p)
    for i in range(4):
        for key in my_data:
            print(f"{my_data[key][0].fetch()[i]*100:.2f}" +" $\pm$ " + f"{my_data[key][1].fetch()[i]*100:.2f}")
        print()

def best_result_metric(data, args):
    return args.A * data[0] - args.B * data[1] + args.A * data[2] + args.C * data[3]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='./')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    lol1 = ""
    lol2 = ""
    res = read_file_data(args.dir)

if __name__ == "__main__":
    main()
