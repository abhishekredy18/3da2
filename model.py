from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            pass
            # TODO:
            # self.decoder =     
            self.decoder = nn.Sequential(
                nn.Linear(512, 8192),
                nn.BatchNorm1d(8192),
                nn.ReLU(),
                nn.Unflatten(1, (128, 4, 4, 4)),
                nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),         
            )
      
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            # self.decoder =  
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, self.n_point * 3)
            )         

        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            # self.decoder =   
            #           
            self.n_verts = mesh_pred.verts_packed().shape[0]
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, self.n_verts * 3),
                nn.Tanh()
            )


    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            # voxels_pred =  
            voxels_pred = self.decoder(encoded_feat)           
            return voxels_pred

        elif args.type == "point":
            # TODO:
            # pointclouds_pred =             
            pointclouds_pred = self.decoder(encoded_feat).view(-1, self.n_point, 3)
            return pointclouds_pred
            
        elif args.type == "mesh":
            # TODO:
            # deform_vertices_pred =     
            deform_vertices_pred = self.decoder(encoded_feat)    
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

