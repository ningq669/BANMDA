import dgl
import copy
import numpy as np
import torch as th
import time
import gc
from utils import EdgeDataset
from minimda import SupernodeLearn
from sklearn import metrics
import torch.utils.data.dataloader as DataLoader
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os
import tqdm
import networkx as nx


#_____________________________________________________
def k_matrix(matrix, k=20):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)
#_____________________________________________________
def get_metrics(score, label):
    y_pre = score
    y_true = label
    metric = caculate_metrics(y_pre, y_true)

    return metric
def caculate_metrics(pre_score, real_score):
    y_pre = pre_score
    y_true = real_score

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision_u, recall_u, thresholds_u = metrics.precision_recall_curve(y_true, y_pre)
    aupr = metrics.auc(recall_u, precision_u)

    y_score = [0 if j < 0.5 else 1 for j in y_pre]

    acc = metrics.accuracy_score(y_true, y_score)
    f1 = metrics.f1_score(y_true, y_score)
    recall = metrics.recall_score(y_true, y_score)
    precision = metrics.precision_score(y_true, y_score)

    metric_result = [auc, aupr, acc, f1, recall, precision]
    print("One epoch metric： ")
    print_met(metric_result)

    return metric_result
def print_met(list):
    print('AUC ：%.4f ' % (list[0]),
          'AUPR ：%.4f ' % (list[1]),
          'Accuracy ：%.4f ' % (list[2]),
          'f1_score ：%.4f ' % (list[3]),
          'recall ：%.4f ' % (list[4]),
          'precision ：%.4f \n' % (list[5]))
def train_test(simData, train_data,args, state):
    os.environ['DGLBACKEND'] = 'pytorch'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    valid_metric = []
    l_n = simData['lncrna_number']
    train_edges = train_data['train_Edges']
    train_labels = train_data['train_Labels']
    test_edges = train_data['test_Edges']
    test_labels = train_data['test_Labels']
    m_d_matrix = train_data['true_md']
    md_class = train_data['md_class']
    ml = simData['ml']
    ld = simData['ld']
    m_d_matrix[tuple(test_edges.T)] = 0#这样做的目的可能是将测试边的权重设置为0，以便在后续的计算中忽略它们。
    md_class[tuple(test_edges.T)] = 0
    #************************************
    ms = simData['ms']
    ds = simData['ds']
    #************************************


    kfolds = args.kfold
    th.manual_seed(42)

    if state == 'valid':
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)
        train_idx, valid_idx = [], []

        for train_index, valid_index in kf.split(train_edges):
            train_idx.append(train_index)
            valid_idx.append(valid_index)

        for i in tqdm.tqdm(range(kfolds)):
            a = i+1
            model = SupernodeLearn(args).cuda()
            optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#            model.load_state_dict(th.load('./cross_valid_example/fold_{}.pkl'.format(a)))
            print(f'################Fold {i + 1} of {kfolds}################')
            edges_train, edges_valid = train_edges[train_idx[i]], train_edges[valid_idx[i]]
            labels_train, labels_valid = train_labels[train_idx[i]], train_labels[valid_idx[i]]
            trainEdges = EdgeDataset(edges_train, labels_train)
            validEdges = EdgeDataset(edges_valid, labels_valid)
            trainLoader = DataLoader.DataLoader(trainEdges, batch_size=args.batchSize, shuffle=True, num_workers=0)
            validLoader = DataLoader.DataLoader(validEdges, batch_size=args.batchSize, shuffle=True, num_workers=0)

            m_d_class = md_class.copy()
            m_d_class[tuple(edges_valid.T)] = 0.0
            md_m = th.from_numpy(m_d_class).cuda()
            md_copy = copy.deepcopy(train_data['train_Edges'])
            md_graph = dgl.graph(
                (
                    np.concatenate((md_copy[:, 0], md_copy[:, 1])), np.concatenate((md_copy[:, 1], md_copy[:, 0]))),
                num_nodes=args.miRNA_number + args.disease_number)
            md_graph = md_graph.to(th.device('cuda:0'))
            ml_copy = copy.deepcopy(ml)
            ml_copy[:, 1] = ml_copy[:, 1] + args.miRNA_number
            ml_graph = dgl.graph(
                (
                    np.concatenate((ml_copy[:, 0], ml_copy[:, 1])), np.concatenate((ml_copy[:, 1], ml_copy[:, 0]))),
                num_nodes=args.miRNA_number + args.lncrna_number)
            ml_graph = ml_graph.to(th.device('cuda:0'))
            ld_copy = copy.deepcopy(ld)
            ld_copy[:, 1] = ld_copy[:, 1] + args.lncrna_number
            ld_graph = dgl.graph(
                (
                    np.concatenate((ld_copy[:, 0], ld_copy[:, 1])), np.concatenate((ld_copy[:, 1], ld_copy[:, 0]))),
                num_nodes=args.lncrna_number + args.disease_number)
            ld_graph = ld_graph.to(th.device('cuda:0'))
            #######################################################
            mm_matrix = k_matrix(ms, 20)
            dd_matrix = k_matrix(ds, 20)
            mm_nx = nx.from_numpy_array(mm_matrix)
            dd_nx = nx.from_numpy_array(dd_matrix)
            mm_graph = dgl.from_networkx(mm_nx)
            dd_graph = dgl.from_networkx(dd_nx)
            mm_graph = mm_graph.to(th.device('cuda:0'))
            dd_graph = dd_graph.to(th.device('cuda:0'))
            ###########################################################

            l_num = l_n


            print("-----training-----")

            for e in tqdm.tqdm(range(args.epoch)):
                running_loss = 0.0
                epo_label = []
                epo_score = []
                print("epoch：", e + 1)
                model.train()
                start = time.time()
                for i, item in enumerate(trainLoader):
                    data, label = item
                    trainData = data.cuda()
                    trainLabel = label.cuda()
                    pre_score = model(simData, md_m, trainData,ml_graph, ld_graph, md_graph, l_num,mm_graph, dd_graph)
                    train_loss = th.nn.BCELoss()
                    loss = train_loss(pre_score, trainLabel)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    running_loss += loss.item()
                    print(f"After batch {i+1}:loss={loss:.3f};",end='\n')
                    batch_score = pre_score.cpu().detach().numpy()
                    epo_score = np.append(epo_score,batch_score)
                    epo_label = np.append(epo_label,label.numpy())
                end=time.time()
                print('Time:%.2f\n'%(end-start))
                train_result = get_metrics(epo_score,epo_label)


            valid_score, valid_label = [], []
            model.eval()
            with th.no_grad():
                print("-----validing-----")
                # val_loss = 0.0
                for i, item in enumerate(validLoader):
                    data, label = item
                    validData = data.cuda()
                    #validLabel = label.cuda()
                    pre_score = model(simData, md_m, validData,ml_graph, ld_graph, md_graph, l_num,)

                    val_batch_score = pre_score.detach().cpu().numpy()
                    valid_score = np.append(valid_score, val_batch_score)
                    valid_label = np.append(valid_label, label.numpy())

                    # torch.save(model.state_dict(), "./fold_{}.pkl".format(a))
                valid_result = get_metrics(valid_score, valid_label)
                valid_metric.append(valid_result)
                gc.collect()
                th.cuda.empty_cache()
        print(np.array(valid_metric))
        cv_metric = np.mean(valid_metric, axis=0)
        print_met(cv_metric)


    else:
        test_score, test_label = [], []
        testEdges = EdgeDataset(test_edges, test_labels)
        testLoader = DataLoader.DataLoader(testEdges, batch_size=args.batchSize, shuffle=False, num_workers=0)
        md_ma = th.from_numpy(md_class).cuda()
        model = SupernodeLearn(args)
        model.load_state_dict(th.load('./test_example/test.pkl'))
        model.cuda()
        model.eval()
        with th.no_grad():
            start = time.time()
            for i, item in enumerate(testLoader):
                data, label = item
                testData = data.cuda()
                pre_score = model(simData, md_ma, testData)
                batch_score = pre_score.cpu().detach().numpy()
                test_score = np.append(test_score, batch_score)
                test_label = np.append(test_label, label.numpy())
            end = time.time()
            print('Time：%.2f \n' % (end - start))
            metrics = get_metrics(test_score, test_label)

    return cv_metric
    # For testing
    # return metrics


def draw_curve(trainloss, trainauc, trainacc, validloss, validauc, validacc, a, b):
    # Curves can be drawn based on train data and test data.
    plt.figure(a + b)
    plt.plot(trainloss, color="blue", label="Train Loss")
    plt.plot(validloss, color="red", label="Valid Loss")
    x_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 99)
    plt.legend(loc='upper right')
    plt.savefig("./curve/loss_{}.png".format(a))

    plt.figure(a + b + 1)
    plt.plot(trainauc, label="Train Auc")
    plt.plot(validauc, color="red", label="Valid Auc")
    x_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 99)
    plt.legend(loc='upper right')
    plt.savefig("./curve_cnn/auc_{}.png".format(a))

    plt.figure(a + b + 2)
    plt.plot(trainacc, color="blue", label="Train Acc")
    plt.plot(validacc, color="red", label="Valid Acc")
    x_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 99)
    plt.legend(loc='upper right')
    plt.savefig("./curve_cnn/acc_{}.png".format(a))