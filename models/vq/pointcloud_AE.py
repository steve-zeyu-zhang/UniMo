import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PointEncoder(nn.Module):
    def __init__(self,args):
        super(PointEncoder,self).__init__()
        self.latent_dim=args.latent_dim
        self.conv1 = torch.nn.Conv1d(args.input_dim, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, args.latent_dim, 1)
    
    def forward(self, xyz):  #[B,L,N,3]->[B,L,latent_dim]
        B,L,N,channel=xyz.shape

        xyz=xyz.contiguous().view(B*L,N,channel)  #[B*L,N,3]
        
        xyz=xyz.permute(0,2,1) #[B*L,3,N]
        
        xyz = F.relu(self.conv1(xyz))
        xyz = F.relu(self.conv2(xyz))
        xyz = self.conv3(xyz)

        # max pooling [B,latent_dim,N]->[B,latent_dim]
        z = torch.max(xyz, dim=2, keepdim=True)[0]
        z = z.view(-1, self.latent_dim)     #[B*L,latent_dim]
        z = z.view(B,L, self.latent_dim)     #[B,L,latent_dim]

        return z
    
class Encoder(nn.Module):
    def __init__(self,args,pointencoder):
        super(Encoder,self).__init__()
        self.latent_dim = args.latent_dim
        self.pointencoder = pointencoder
        
        self.args=args
        

    def forward(self, x):  #[B,L,N,3]->[B,L,latent_dim]

        B,L,N,channel=x.shape
        
        x_encoded = self.pointencoder(x)  #[B,L,N,3]->[B,L,latent_dim]

        x_encoded=x_encoded.permute(1,0,2)   #[B,L,latent_dim] -> [L, B, latent_dim]
        
        return x_encoded


class Transformer_pointdecoder_base_v2(nn.Module):
    def __init__(self, args):
        super(Transformer_pointdecoder_base_v2, self).__init__()
        self.latent_dim = args.latent_dim
        self.k = args.Transformer_pointdecoder_k
        self.num_branch = args.Transformer_pointdecoder_num_branch
        self.n = args.points_num//self.num_branch  

        self.fc_Q = nn.Linear(self.latent_dim, self.n*self.k)
        self.fc_K = nn.Linear(self.latent_dim, self.n*self.k)
        self.fc_V = nn.Linear(self.latent_dim, self.n*self.k)
        self.fc_Ori = nn.Linear(self.latent_dim, self.n*self.k)
        self.softmax = nn.Softmax(dim=2)
        self.conv_output = nn.Conv1d(self.k, 3, 1)
        self.conv_ori = nn.Conv1d(self.k, 3, 1)


    def forward(self, x):
        # B,L,N,channel=batch["xyz"].shape
        # x=batch["point_encoder_z"]
        ori=self.fc_Q(x) #[B,latent_dim]->[B,n*k]  论文的结果
        #ori=self.fc_Ori(x) #[B,latent_dim]->[B,n*k]
        Q = self.fc_Q(x) #[B,latent_dim]->[B,n*k]
        K = self.fc_K(x) #[B,latent_dim]->[B,n*k]
        V = self.fc_V(x) #[B,latent_dim]->[B,n*k]
        ori = ori.view(-1, self.n, self.k)  #[B,n,k]
        Q = Q.view(-1, self.n, self.k)  #[B,n,k]
        K = K.view(-1, self.n, self.k)  #[B,n,k]
        V = V.view(-1, self.n, self.k)  #[B,n,k]
        
        ori=ori.transpose(1, 2).contiguous()  #[B,K,n]
        ori=self.conv_ori(ori)   #[B,3,n]
        
        K= K.transpose(1, 2).contiguous()  #[B,K,n]
        
        #x=self.softmax(torch.div(torch.bmm(Q, K), math.sqrt(self.k)))   #[B,num_points//num_branch,K]
        x=self.softmax(torch.bmm(Q, K)/math.sqrt(self.k))   #[B,n,n]
        x=torch.bmm(x, V)    #[B,n,k]
        x= x.transpose(1, 2).contiguous()  #[B,K,n]
        x=self.conv_output(x)   #[B,3,n]
        x=x+ori
        x= x.transpose(1, 2).contiguous()  #[B,n,3]

        return x
    
class MappingNet(nn.Module):
    def __init__(self, args):
        super(MappingNet, self).__init__()
        self.K1 = args.latent_dim

        self.fc1 = nn.Linear(self.K1, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, self.K1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(self.K1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        return x
    
class Transformer_pointdecoder(nn.Module):
    def __init__(self,args):
        super(Transformer_pointdecoder,self).__init__()

        self.featmap = MappingNet(args)
        self.mapping_net=True

        self.pointgen =Transformer_pointdecoder_base_v2(args)

        
    def forward(self, z):
        output = torch.empty(size=(z.shape[0], 0, 3)).to(z.device)

        _x_1=z

        _x_1=self.featmap(_x_1)
        #print(_x_1.shape)

        _x_1 = self.pointgen(_x_1)
        #print(_x_1.shape)

        output = torch.cat((output, _x_1), dim=1)
        
        return output

class Decoder(nn.Module):
    def __init__(self,args):
        super(Decoder,self).__init__()
        self.latent_dim=args.latent_dim

        self.points_decoder=Transformer_pointdecoder(args)
        
    
    def forward(self, z):  #[B, L, latent_dim] -> [B,L,N,3]

        B, L, latent_dim = z.shape

        z = z.reshape(B*L, latent_dim)
        
        z_decoded = self.points_decoder(z)  #[B*L, latent_dim] -> [B, L, N, 3]

        z_decoded = z_decoded.view(B, L, -1, 3)

        return z_decoded


class PCMGAE(nn.Module):
    def __init__(self,args):
        super(PCMGAE,self).__init__()

        self.pointencoder=PointEncoder(args)

        #self.pointencoder=DGCNN(args)
        self.latent_dim=args.latent_dim
        
        self.encoder=Encoder(args,self.pointencoder)
        self.decoder=Decoder(args)

        
    def forward(self, x):
        
        x_encoded = self.encoder(x)
        
        x_decoded = self.decoder(x_encoded)

        return x_decoded, torch.Tensor([0.0]).to(x.device), torch.Tensor([0.0]).to(x.device)
