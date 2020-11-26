# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from torch import nn
import torch
import math
import graph.ntu_rgb_d
import graph.stgraph

class SGN(nn.Module):
    def __init__(self, num_classes, dataset, seg, args, bias=True):
        super(SGN, self).__init__()

        self.dim1 = 256
        self.dataset = dataset
        self.seg = seg
        num_joint = 25
        bs = args.batch_size
        self.A = graph.ntu_rgb_d.AdjMatrixGraph().A
        self.large_A=graph.stgraph.stAdjGraph().stA
        if args.train:
            self.spa = self.one_hot(bs, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(bs, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()
        else:
            self.spa = self.one_hot(32 * 5, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(32 * 5, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()

        self.tem_embed = embed(self.seg, 64 * 4, norm=False, bias=bias)  # 64*4表示论文中的C3，ft的embedding
        self.spa_embed = embed(num_joint, 64, norm=False, bias=bias)  # C1=64,jk的embedding
        self.data_bn=nn.BatchNorm1d(150)
        self.bn34=nn.BatchNorm2d(256)
        # self.joint_embed = embed(3, 64, norm=True, bias=bias)     #ptk的embedding
        self.doublejoint_embed = embed(6, 64, norm=False, bias=bias)  # 双人
        # self.dif_embed = embed(3, 64, norm=True, bias=bias)       #vtk的embedding
        self.doubledif_embed = embed(6, 64, norm=False, bias=bias)  # 双人
        #self.norm=nn.BatchNorm2d(6)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)  # TMP前的两个CNN
        #self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.compute_LG = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        ###adagcn
        self.gcn1_1 = largegcn(128, 128, bias=bias)
        self.gcn1_2 = largegcn(128, 256, bias=bias)
        self.gcn1_3=largegcn(256,256,bias=bias)
        ###fixgcn
        self.gcn2_1 = largegcn(128, 128, bias=bias)
        self.gcn2_2 = largegcn(128, 256, bias=bias)
        self.gcn2_3=largegcn(256,256,bias=bias)
        ###comgcn
        self.gcn3_1 = puregcn(128, 128, bias=bias)
        self.gcn3_2 = puregcn(128, 256, bias=bias)
        self.gcn3_3=puregcn(256,256,bias=bias)
        self.fc = nn.Linear(self.dim1 * 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))      #用正态分布初始化

        nn.init.constant_(self.gcn1_1.w.cnn.weight, 0)   #W的权值初始化为0
        nn.init.constant_(self.gcn1_2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn1_3.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2_1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2_2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2_3.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3_1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3_2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3_3.w.cnn.weight, 0)

    def forward(self, input):

        # Dynamic Representation
        bs, step, dim = input.size()  # 64*20*150
        num_joints = 25
        input = input.permute(0, 2, 1).contiguous()
        input = self.data_bn(input)
        input = input.permute(0, 2, 1).contiguous()
        input = input.view((bs, step, num_joints, 6)).contiguous()  # 64*20*# 25*6
        input = input.permute(0, 3, 2, 1).contiguous()
        dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
        dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)  # 最后一帧速度为0
        Linput = input.view((bs, 6, step * num_joints, 1)).contiguous()
        Ldif = dif.view((bs, 6, step * num_joints, 1)).contiguous()
        Ldif = self.doubledif_embed(Ldif)
        Lpos = self.doublejoint_embed(Linput)
        if Lpos.shape[0] < self.tem.shape[0]:
            tem11 = self.tem_embed(self.tem)
            spa11 = self.spa_embed(self.spa)
            tem1 = tem11[0:input.shape[0], :, :, :]
            spa1 = spa11[0:input.shape[0], :, :, :]
        else:
            tem1 = self.tem_embed(self.tem)
            spa1 = self.spa_embed(self.spa)  # 64*64*25*20

        dy = Lpos + Ldif
        # Joint-level Module
        spa1=spa1.view(bs,64,num_joints*step,1).contiguous()
        #dy=dy.view(bs,64,num_joints,step).contiguous()
        Linput = torch.cat([dy, spa1], 1)  # 64*128*25*20
        #Linput=input.view(bs,128,num_joints*step,1).contiguous()
        #input=input.view(bs,128,num_joints*step,1).contiguous()
        LG=self.compute_LG(Linput)
        #g = self.compute_g1(input)  # 论文中的G，T*J*J
        A=torch.from_numpy(self.large_A).cuda()
        A=A.unsqueeze(0)
        A = A.unsqueeze(0)
        A = A.repeat(bs, 1, 1,1)
        input1 = self.gcn1_1(Linput, LG)
        input1 = self.gcn1_2(input1, LG)
        input1 = self.gcn1_3(input1, LG)
        input1=input1.view(bs, 256, num_joints, step).contiguous()
        input2=self.gcn2_1(Linput,A)
        input2 = self.gcn2_2(input2, A)
        input2 = self.gcn2_3(input2, A)
        input2 = input2.view(bs, 256, num_joints, step).contiguous()
        input3=self.gcn3_1(Linput,A)
        input3 = self.gcn3_2(input3, A)
        input3 = self.gcn3_3(input3, A)
        input3 = input3.view(bs, 256, num_joints, step).contiguous()
        input4 = self.gcn3_1(Linput,LG)
        input4 = self.gcn3_2(input4, LG)
        input4 = self.gcn3_3(input4, LG)
        input4 = input4.view(bs, 256, num_joints, step).contiguous()
        input3=self.bn34((input3+input4)/2)
        #input=self.testcnn(input)
        #input = input.view(bs, 256, num_joints, step)
        #Binput=(input3+input4)/2
        #input=torch.stack([input1,input2,Binput],1)
        #bs,N,fea,joint,T=input.size()
        #input=input.view(bs,N,fea,joint*T)
        ##attrntion
        #input=input.permute(0,2,1,3)
        #W=self.attpro(input)
        #beta=torch.softmax(W,2)
        #beta=beta.permute(0, 2, 3, 1)  #64*3*500*1
        #input=input.permute(0,2,3,1)
        #input=input*beta
        #input=input.sum(1)
        #input=torch.squeeze(input)
        #input=input.view(bs,joint,T,fea)
        #input=input.permute(0,3,1,2)
        Linput=(input1+input2+input3)/3
        # Frame-level Module
        #input1=input1.view(bs,256,num_joints,step).contiguous()
        Linput = Linput + tem1
        Linput = self.cnn(Linput)
        # Classification
        output = self.maxpool(Linput)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output


    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot


class norm_data(nn.Module):
    def __init__(self, dim=64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim * 25)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x


class embed(nn.Module):
    def __init__(self, dim=3, dim1=128, norm=True, bias=False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x


class cnn1x1(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x

class local(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        x1 = self.maxpool(x1)
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class puregcn(nn.Module):
    def __init__(self, in_feature, out_feature, bias=False):
        super(puregcn, self).__init__()
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)

    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = torch.matmul(g,x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(x)
        return x

class largegcn(nn.Module):
    def __init__(self, in_feature, out_feature, bias=False):
        super(largegcn, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)

    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = torch.matmul(g,x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        N, outfeature, TJ,_ = x.size()
        x=x.view(N,outfeature,25,20).contiguous()
        x = self.relu(self.bn(x))
        x=x.view(N,outfeature,TJ,1).contiguous()
        return x


class compute_g_spa(nn.Module):
    def __init__(self, dim1=64 * 3, dim2=64 * 3, bias=False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):
        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g

