import torch as th
from torch import nn
from dgl import function as fn
from otherlayers import *
from extractSubGraph import GetSubgraph
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ConvLayer(nn.Module):

    def __init__(self, in_feats, out_feats, k=2, method='sum', bias=True, batchnorm=False, activation='relu',
                 dropout=0.0):
        super(ConvLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.k = k + 1
        self.method = method
        self.weights = []
        for i in range(self.k):
            self.weights.append(nn.Parameter(th.Tensor(in_feats, out_feats).to('cuda')))
        self.biases = None
        self.activation = None
        self.batchnorm = None
        self.dropout = None

        if bias:
            self.biases = []
            for i in range(self.k):
                self.biases.append(nn.Parameter(th.Tensor(out_feats).to('cuda')))

        self.reset_parameters()

        if activation == 'relu':
            self.activation = th.relu
        if batchnorm:
            if method == 'cat':
                self.batchnorm = nn.BatchNorm1d(out_feats * self.k)
            else:
                self.batchnorm = nn.BatchNorm1d(out_feats)
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for i in range(self.k):
            nn.init.xavier_uniform_(self.weights[i])
            if self.biases is not None:
                nn.init.zeros_(self.biases[i])

    def forward(self, graph, feat):
        feat = feat.to('cuda')
        with graph.local_scope():
            degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp)

            if self.biases is not None:
                rst = th.matmul(feat, self.weights[0]) + self.biases[0]
            else:
                rst = th.matmul(feat, self.weights[0])

            for i in range(1, self.k):
                feat = feat * norm

                graph.ndata['h'] = feat
                if 'e' in graph.edata.keys():
                    graph.update_all(fn.u_mul_e('h', 'e', 'm'), fn.sum('m', 'h'))
                else:
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                feat = graph.ndata.pop('h')
                feat = feat * norm


                if self.method == 'sum':
                    if self.biases is not None:
                        y = th.matmul(feat, self.weights[0]) + self.biases[0]
                    else:
                        y = th.matmul(feat, self.weights[0])
                    rst = rst + y
                elif self.method == 'mean':
                    if self.biases is not None:
                        y = th.matmul(feat, self.weights[0]) + self.biases[0]

                    else:
                        y = th.matmul(feat, self.weights[0])
                    rst = rst + y
                    rst = rst / self.k
                elif self.method == 'cat':
                    if self.biases is not None:
                        y = th.matmul(feat, self.weights[0]) + self.biases[0]
                    else:
                        y = th.matmul(feat, self.weights[0])
                    rst = th.cat((rst, y), dim=1)

            if self.batchnorm is not None:
                rst = self.batchnorm(rst)
            if self.activation is not None:
                rst = self.activation(rst)
            if self.dropout is not None:
                rst = self.dropout(rst)
            return rst


class GraphEmbbeding(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, k, method, bias, batchnorm, activation, num_layers, dropout):
        super(GraphEmbbeding, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                hid_feats = out_feats
            self.layers.append(ConvLayer(in_feats, hid_feats, k, method, bias, batchnorm, activation, dropout))
            if method == 'cat':
                in_feats = hid_feats * (k + 1)
            else:
                in_feats = hid_feats

    def forward(self, graph, feat):
        feat = feat.to('cuda')
        for i, layer in enumerate(self.layers):
            feat = layer(graph, feat)
        return feat

class SimMatrix(nn.Module):
    def __init__(self, args):
        super(SimMatrix, self).__init__()
        self.mnum = args.m_num
        self.dnum = args.d_num
        self.viewn = args.view
        self.attsim_m = SimAttention(self.mnum, self.mnum, self.viewn)
        self.attsim_d = SimAttention(self.dnum, self.dnum, self.viewn)

    def forward(self, data):
        m_funsim = data['mm_f'].cuda()
        m_seqsim = data['mm_s'].cuda()
        m_gossim = data['mm_g'].cuda()
        d_funsim = data['dd_t'].cuda()
        d_semsim = data['dd_s'].cuda()
        d_gossim = data['dd_g'].cuda()

        m_sim = th.stack((m_funsim, m_seqsim, m_gossim), 0)
        d_sim = th.stack((d_funsim, d_semsim, d_gossim), 0)
        m_attsim = self.attsim_m(m_sim)
        d_attsim = self.attsim_d(d_sim)

        # Set the diagonal to 0.0 for subsequent sampling.
        m_final_sim = m_attsim.fill_diagonal_(fill_value=0)
        d_final_sim = d_attsim.fill_diagonal_(fill_value=0)

        return m_final_sim, d_final_sim


class SupernodeLearn(nn.Module):
    def __init__(self, args):
        super(SupernodeLearn, self).__init__()

        self.hop = args.hop
        self.neigh_size = args.nei_size
        self.mNum = args.m_num
        self.dNum = args.d_num
        self.simClass = args.sim_class
        self.mdClass = args.md_class
        self.class_all = self.simClass + self.simClass + self.mdClass
        self.NodeFea = args.feture_size
        self.hinddenSize = args.atthidden_fea
        self.edgeFea = args.edge_feature
        self.drop = args.Dropout

        self.actfun = nn.LeakyReLU(negative_slope=0.2)
        self.actfun2 = nn.Sigmoid()

        self.SimGet = SimMatrix(args)

        self.edgeTran = OnehotTran(self.simClass, self.mdClass, self.mNum, self.dNum)
        self.getSubgraph = GetSubgraph(self.neigh_size, self.hop)

        self.EMBnode = NodeEmbedding(self.mNum, self.dNum, self.NodeFea, self.drop)
        self.EMBedge = EdgeEmbedding(self.simClass, self.mdClass, self.neigh_size)

        self.NeiAtt = NeiAttention(self.edgeFea, self.NodeFea, self.neigh_size)

        self.Agg = NeiAggregator(self.NodeFea, self.drop, self.actfun)

        self.fcLinear1 = MLP(768, 1, self.drop, self.actfun)
        #self.fcLinear2 = MLP(128, 1, self.drop, self.actfun)
        self.lin_m = nn.Linear(args.m_num, args.in_feats, bias=False)
        self.lin_d = nn.Linear(args.d_num, args.in_feats, bias=False)
        self.args = args
        self.gcn_md = GraphEmbbeding(args.in_feats, args.hid_feats, args.out_feats, args.k, args.method, args.gcn_bias,
                                 args.gcn_batchnorm, args.gcn_activation, args.num_layers, args.dropout)
        self.gcn_ml = GraphEmbbeding(args.in_feats, args.hid_feats, args.out_feats, args.k, args.method, args.gcn_bias,
                                 args.gcn_batchnorm, args.gcn_activation, args.num_layers, args.dropout)
        self.gcn_ld = GraphEmbbeding(args.in_feats, args.hid_feats, args.out_feats, args.k, args.method, args.gcn_bias,
                                 args.gcn_batchnorm, args.gcn_activation, args.num_layers, args.dropout)
#################################################################消融实验#########################################################################################
        self.gcn_mm = GraphEmbbeding(args.m_num, args.hid_feats, args.out_feats, args.k, args.method,
                                     args.gcn_bias,
                                     args.gcn_batchnorm, args.gcn_activation, args.num_layers, args.dropout)
        self.gcn_dd = GraphEmbbeding(args.d_num, args.hid_feats, args.out_feats, args.k, args.method,
                                     args.gcn_bias,
                                     args.gcn_batchnorm, args.gcn_activation, args.num_layers, args.dropout)



        self.BilinearDecoder = BilinearDecoder(feature_size=64)
        self.h_fc = nn.Linear(384, 64)

    def forward(self, simData, m_d, md_node,ml_graph, ld_graph, md_graph, l_num,mm_graph,dd_graph):
        # Get the similarity.
        m_sim, d_sim = self.SimGet(simData)
        # The original association matrix can be used for edge selection.
        prep_one = th.cat((m_sim, m_d), dim=1)
        prep_two = th.cat((m_d.t(), d_sim), dim=1)
        md_all = th.cat((prep_one, prep_two), dim=0)

        # Redefine the index of the node.
        m_node = md_node[:, 0]
        d_node = md_node[:, 1] + self.mNum

        relation_adj = self.edgeTran(m_sim, d_sim, m_d)
        # Subgraph extraction.
        m_neinode_list, m_neirel_list, d_neinode_list, d_neirel_list = self.getSubgraph(m_node, d_node, md_all,
                                                                                        relation_adj)

        # Get embedding representation.
        m_nodeemb_list = self.EMBnode(m_sim, d_sim, m_neinode_list)
        d_nodeemb_list = self.EMBnode(m_sim, d_sim, d_neinode_list)
        m_relemb_list = self.EMBedge(m_neirel_list)
        d_relemb_list = self.EMBedge(d_neirel_list)
        # Gather long distance information by node-aware attention.
        for i in range(self.hop - 1, 0, -1):
            mneigh_update_emb = self.NeiAtt(m_nodeemb_list[i], m_relemb_list[i], m_nodeemb_list[i + 1], i)
            dneigh_update_emb = self.NeiAtt(d_nodeemb_list[i], d_relemb_list[i], d_nodeemb_list[i + 1], i)

            m_nodeemb_list[i] = self.Agg(m_nodeemb_list[i], mneigh_update_emb)
            d_nodeemb_list[i] = self.Agg(d_nodeemb_list[i], dneigh_update_emb)
        m_nodeemb=m_nodeemb_list[0]
        d_nodeemb=d_nodeemb_list[0]
        m_nodeemb=m_nodeemb.view(m_nodeemb.size(0), -1)
        d_nodeemb = d_nodeemb.view(d_nodeemb.size(0), -1)
        # Hyperedge-aware attention aggregates direct neighborhood information to learn and identification.、


        #md_n = th.randn((md_num, self.args.in_feats))
        l_n = th.rand((l_num, self.args.in_feats)).to(th.device('cuda:0'))
        #ld_n = th.randn((ld_num, self.args.in_feats))
        ##############################################################
        emb_mm_sim = self.gcn_mm(mm_graph, m_sim)
        emb_dd_sim = self.gcn_dd(dd_graph, d_sim)
        ##########################################################
        emb_ass2 = self.gcn_ml(ml_graph, th.cat((self.lin_m(m_sim),l_n), dim=0))
        emb_ass = self.gcn_md(md_graph, th.cat((self.lin_m(m_sim),self.lin_d(d_sim)), dim=0))
        emb_ass3 = self.gcn_ld(ld_graph, th.cat((l_n,self.lin_d(d_sim)), dim=0))
        emb_mm_ass = emb_ass[:self.args.miRNA_number, :]
        emb_dd_ass = emb_ass[self.args.miRNA_number:, :]
        emb_mm_ass2 = emb_ass2[:self.args.miRNA_number, :]
        emb_dd_ass2 = emb_ass3[self.args.lncrna_number:, :]
        emb_mm = th.cat((emb_mm_ass, emb_mm_ass2,emb_mm_sim), dim=1)
        emb_dd = th.cat((emb_dd_ass, emb_dd_ass2,emb_dd_sim), dim=1)
        #*************************************
        #emb_mm =emb_mm[md_node[:, 0]]
       # emb_dd =emb_dd[md_node[:, 1]]
       # emb1 =emb_mm + m_nodeemb
        #emb2 =emb_dd + d_nodeemb
        #emb = emb1*emb2
        # *************************************
        emb1= th.cat((emb_mm[md_node[:, 0]], emb_dd[md_node[:, 1]]), dim=1)
        emd2= th.cat((m_nodeemb, d_nodeemb), dim=1)
        emb=th.cat(( emb1,emd2),dim=1)
        f_mirnas = th.cat((m_nodeemb, emb_mm[md_node[:, 0]]), dim=1)
        f_diseases = th.cat((d_nodeemb, emb_dd[md_node[:, 1]]), dim=1)


        h_mirnas = self.actfun(self.h_fc(emb_mm[md_node[:, 0]]))
        h_diseases = self.actfun(self.h_fc(emb_mm[md_node[:, 0]]))


        pre_score = self.BilinearDecoder(h_diseases, h_mirnas)


        return pre_score
class BilinearDecoder(nn.Module):
    def __init__(self, feature_size):
        super(BilinearDecoder, self).__init__()
        self.W = Parameter(torch.randn(feature_size, feature_size))

    def forward(self, h_diseases, h_mirnas):
        h_diseases0 = torch.mm(h_diseases, self.W)
        h_mirnas0 = torch.mul(h_diseases0, h_mirnas)
        h0 = h_mirnas0.sum(1)
        h = torch.sigmoid(h0)
        return h

# 内积解码器
class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, h_diseases, h_mirnas):
        x = torch.mul(h_diseases, h_mirnas).sum(1)
        x = torch.reshape(x, [-1])
        outputs = F.sigmoid(x)
        return outputs