import numpy as np
from argparse import ArgumentParser
from cifar_data import *
from model import Model

DEFAULT_OUTFILE = 'result.csv'
DEFAULT_CIFAR_PATH = 'datasets/cifar-100-python'
alpha = 1e-3
beta = 1e1
gamma = 1e-5
verbose = 50
types = ["fruit_and_vegetables", "household_electrical_devices"]

def get_problems(data):
    problems = []
    for pos1 in data[types[0]]:
        for pos2 in data[types[0]]:
            if pos1 == pos2:
                continue
            for neg1 in data[types[1]]:
                for neg2 in data[types[1]]:
                    if neg1 == neg2:
                        continue
                    name = "%s-%s-vs-%s-%s"%(pos1, neg1, pos2, neg2)
                    problems.append(name)
    return problems

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--cifar_path', dest='cifar_path', metavar='CIFAR_PATH',
        help='directory of CIFAR-100 dataset', default=DEFAULT_CIFAR_PATH)
    parser.add_argument('--outfile', dest='outfile', metavar='OUTPUTFILE',
        help='file that save the result', default=DEFAULT_OUTFILE)
    return parser

def main():
    parser = get_parser()
    option = parser.parse_args()
    assert os.path.isdir(option.cifar_path), "You may forget to download CIFAR-100 "\
                                        "dataset, or please specify the path "\
                                        "by --cifar_path option if you have the "\
                                        "dataset already but not in default path %s."%DEFAULT_CIFAR_PATH
    data = load_data(types, option.cifar_path)
    problems = get_problems(data)
    rr = np.zeros((len(problems), ))
    for i, name in enumerate(problems):
        pos1, neg1, _, pos2, neg2 = name.split('-')
        Xs, ys = get_XY(data[types[0]][pos1], data[types[1]][neg1])
        Xt, yt = get_XY(data[types[0]][pos2], data[types[1]][neg2])
        src = data_holder(Xs, ys)
        targ = data_holder(Xt, yt)
        print 'Train classifier for ', name
        print 'alpha %.6f, beta %.6f, gamma %.6f'%(alpha, beta, gamma)
        with Model() as model:
            model.build()
            model.inference(alpha, beta, gamma)
            model.train(src, targ, verbose=verbose)
            rr[i] = model.test(targ)
        print 'Test Accuracy is ', rr[i]
        with open(option.outfile, 'wb') as f:
            np.savetxt(f, rr, delimiter=',', fmt='%.4f')
    print 'The result is stored in %s and mean accuracy is %.4f'%(option.outfile, np.mean(rr, axis=0))

if __name__ == '__main__':
    main()
