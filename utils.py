
import random
import torch as th
import numpy as np
import csv
import torch.utils.data.dataset as Dataset

def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return th.Tensor(md_data)


def Simdata_processing(args):
    dataset = dict()

    mm_funsim = read_csv(args.datapath + '/m_fs.csv')
    dataset['mm_f'] = mm_funsim

    mm_seqsim = read_csv(args.datapath + '/m_ss.csv')
    dataset['mm_s'] = mm_seqsim

    mm_gausim = read_csv(args.datapath + '/m_gs.csv')
    dataset['mm_g'] = mm_gausim

    dd_funsim = read_csv(args.datapath + '/d_ts.csv')
    dataset['dd_t'] = dd_funsim

    dd_semsim = read_csv(args.datapath + '/d_ss.csv')
    dataset['dd_s'] = dd_semsim

    dd_gausim = read_csv(args.datapath + '/d_gs.csv')
    dataset['dd_g'] = dd_gausim
    return dataset
def Simdata_pro(args):
    dataset = dict()
    mm_funsim = np.loadtxt(args.datapath + 'm_fs.csv', dtype=np.float64, delimiter=',')
    mm_seqsim = np.loadtxt(args.datapath + 'm_ss.csv', dtype=np.float64, delimiter=',')
    mm_gausim = np.loadtxt(args.datapath + 'm_gs.csv', dtype=np.float64, delimiter=',')
    dd_funsim = np.loadtxt(args.datapath + 'd_ts.csv', dtype=np.float64, delimiter=',')
    dd_semsim = np.loadtxt(args.datapath + 'd_ss.csv', dtype=np.float64, delimiter=',')
    dd_gausim = np.loadtxt(args.datapath + 'd_gs.csv', dtype=np.float64, delimiter=',')

    dataset['mm_f'] = th.FloatTensor(mm_funsim)
    dataset['mm_s'] = th.FloatTensor(mm_seqsim)
    dataset['dd_t'] = th.FloatTensor(dd_funsim)
    dataset['dd_s'] = th.FloatTensor(dd_semsim)
    dataset['mm_g'] = th.FloatTensor(mm_gausim)
    dataset['dd_g'] = th.FloatTensor(dd_gausim)
    m = np.loadtxt(args.data_dir + 'miRNA number.txt', delimiter='\t', dtype=np.str_)
    d = np.loadtxt(args.data_dir + 'disease number.txt', delimiter='\t', dtype=np.str_)
    l = np.loadtxt(args.data_dir + 'lncrna number.txt', delimiter='\t', dtype=np.str_)

    dataset['miRNA_number'] = int(m.shape[0])
    dataset['disease_number'] = int(d.shape[0])
    dataset['lncrna_number'] = int(l.shape[0])

    dataset['d_num'] = np.loadtxt(args.data_dir + 'disease number.txt', delimiter='\t', dtype=np.str_)[:, 1]
    dataset['m_num'] = np.loadtxt(args.data_dir + 'miRNA number.txt', delimiter='\t', dtype=np.str_)[:, 1]
    dataset['l_num'] = np.loadtxt(args.data_dir + 'lncrna number.txt', delimiter='\t', dtype=np.str_)[:, 1]
    dataset['md'] = np.loadtxt(args.data_dir + 'known disease-miRNA association number.txt', dtype=np.int32) - 1
    dataset['ml'] = np.loadtxt(args.data_dir + 'known lncrna-miRNA association number.txt', dtype=np.int32) - 1
    dataset['ld'] = np.loadtxt(args.data_dir + 'known disease-lncrna association number.txt', dtype=np.int32) - 1




    dataset['ms'] = (dataset['mm_f']+ dataset['mm_s'] + dataset['mm_g'])/3
    dataset['ds'] = (dataset['dd_t']+dataset['dd_s']+dataset['dd_g'])/3


    return dataset
def load_data(args):
    # Load the original miRNA-disease associations matrix.
    md_matr = np.loadtxt(args.datapath + '/m_d.csv', dtype=np.float32, delimiter=',')

    rng = np.random.default_rng(seed=42)
    pos_samples = np.where(md_matr == 1)

    pos_samples_shuffled = rng.permutation(pos_samples, axis=1)

    rng = np.random.default_rng(seed=42)
    neg_samples = np.where(md_matr == 0)
    neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :pos_samples_shuffled.shape[1]]

    edge_idx_dict = dict()
    n_pos_samples = pos_samples_shuffled.shape[1]#正样本数量
    idx_split = int(n_pos_samples * args.ratio)#五折交叉

    test_pos_edges = pos_samples_shuffled[:, :idx_split]#测试
    test_neg_edges = neg_samples_shuffled[:, :idx_split]
    test_pos_edges = test_pos_edges.T#转置
    test_neg_edges = test_neg_edges.T
    test_true_label = np.hstack((np.ones(test_pos_edges.shape[0]), np.zeros(test_neg_edges.shape[0])))#标签水平方向堆叠
    test_true_label = np.array(test_true_label, dtype='float32')
    test_edges = np.vstack((test_pos_edges, test_neg_edges))#边垂直方向堆叠

    train_pos_edges = pos_samples_shuffled[:, idx_split:]
    train_neg_edges = neg_samples_shuffled[:, idx_split:]
    train_pos_edges = train_pos_edges.T
    train_neg_edges = train_neg_edges.T
    train_true_label = np.hstack((np.ones(train_pos_edges.shape[0]), np.zeros(train_neg_edges.shape[0])))
    train_true_label = np.array(train_true_label, dtype='float32')
    train_edges = np.vstack((train_pos_edges, train_neg_edges))
    edge_idx_dict['train_Edges'] = train_edges
    edge_idx_dict['train_Labels'] = train_true_label
    edge_idx_dict['test_Edges'] = test_edges
    edge_idx_dict['test_Labels'] = test_true_label

    # Load the collected miRNA-disease associations matrix with edge attributes.
    md_class = np.loadtxt(args.datapath + '/m_d_edge.csv', dtype=np.float32, delimiter=',')
    edge_idx_dict['md_class'] = md_class
    edge_idx_dict['true_md'] = md_matr

    return edge_idx_dict
class EdgeDataset(Dataset.Dataset):
    def __init__(self, edges, labels):
        self.Data = edges
        self.Label = labels

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label