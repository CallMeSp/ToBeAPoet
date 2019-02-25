import argparse


def getConfig_Train(args=''):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--epoch_num', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    args = parser.parse_args(args.split())
    return args
