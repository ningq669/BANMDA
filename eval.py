import argparse
from utils import Simdata_pro, load_data
from train import train_test
def result(args):
    simData = Simdata_pro(args)
    args.miRNA_number = simData['miRNA_number']
    args.disease_number = simData['disease_number']
    args.lncrna_number = simData['lncrna_number']
    train_data = load_data(args)
    results = train_test(simData, train_data,args, state='valid')


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1, metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument("--in_feats", type=int, default=1024, help='Input layer dimensionalities.')
parser.add_argument("--hid_feats", type=int, default=128, help='Hidden layer dimensionalities.')
parser.add_argument("--out_feats", type=int, default=128, help='Output layer dimensionalities.')
parser.add_argument("--method", default='sum', help='Merge feature method')
parser.add_argument("--gcn_bias", type=bool, default=True, help='gcn bias')
parser.add_argument("--gcn_batchnorm", type=bool, default=True, help='gcn batchnorm')
parser.add_argument("--gcn_activation", default='relu', help='gcn activation')
parser.add_argument("--num_layers", type=int, default=2, help='Number of GNN layers.')
parser.add_argument('--k', type=int, default=4, help='k order')
parser.add_argument('--dropout', type=float, default=0, help='gcn dropout')
parser.add_argument('--dataset', default='HMDD v3.2', help='dataset')

parser.add_argument('--datapath', default='./dataset/', help='dataset')
parser.add_argument('--ratio', type=float,default=0.2, help='ratio')
parser.add_argument('--kfold', type=int,default=5, help='kfold')
parser.add_argument('--nei_size', type=list,default=[256, 32], help='nei_size')
parser.add_argument('--hop', type=int,default=2, help='hop')
parser.add_argument('--feture_size', type=int,default=256, help='feture_size')
parser.add_argument('--edge_feature', type=int,default=9, help='edge_feature')
parser.add_argument('--atthidden_fea', type=int,default=128, help='atthidden_fea')
parser.add_argument('--sim_class', type=int,default=3, help='sim_class')
parser.add_argument('--md_class', type=int,default=3, help='md_class')
parser.add_argument('--m_num', type=int,default=853, help='m_num')
parser.add_argument('--d_num', type=int,default=591, help='d_num')
parser.add_argument('--batchSize', type=int,default=64, help='batchSize')
parser.add_argument('--Dropout', type=float, default=0, help='Dropout')
parser.add_argument('--view', type=int,default=3, help='view')
parser.add_argument('--weight_decay', type=float,default=0.0005, help='weight_decay')

args = parser.parse_args()
args.data_dir = 'data/' + args.dataset + '/'
args.result_dir = 'result/' + args.dataset + '/'

result(args)
