import numpy as np
import argparse
import os

def read_using_keyword(line:str, keyword:str, offset:int):
    index = line.find(keyword) + len(keyword)
    return line[index:index+offset]

def read_file_data(file_data, distance):
    for line in file_data:
        if "Dist: " in line:
            this_dist = float(read_using_keyword(line, "Dist: ", 6))
            if this_dist > distance:
                break
            acc = float(read_using_keyword(line, "Acc: ", 6))
            asr = float(read_using_keyword(line, "ASR: ", 6))
    return acc, asr

def best_result_metric(data, args):
    return args.A * data[0] - args.B * data[1] + args.A * data[2] + args.C * data[3]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='./')
    args = parser.parse_args()
    return args

def main():
    distances = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]
    args = parse_args()
    lol1 = ""
    lol2 = ""
    for dist in distances:
        acc = []
        asr = []
        for i in range(1,4):
            fn = f"{args.dir}.{i}"
            with open(fn) as f:
                file_data = f.read().splitlines()
            res = read_file_data(file_data, dist)
            acc.append(res[0])
            asr.append(res[1])
        print(f"Dist: {dist:.3f}, acc: {np.mean(acc):.4f}, asr: {np.mean(asr):.4f}")
        lol1 += f"{np.mean(acc):.4f}\n"
        lol2 += f"{np.mean(asr):.4f}\n"
    print(lol1, lol2)

if __name__ == "__main__":
    main()