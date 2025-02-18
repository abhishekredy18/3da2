import os
import pickle
import imageio
import argparse
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import torch
import mcubes

import pytorch3d
from pytorch3d.io import load_obj

from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)

from pytorch3d.structures import Meshes
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import TexturesVertex

def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def unproject_depth_image(image, mask, depth, camera):
    """
    Unprojects a depth image into a 3D point cloud.

    Args:
        image (torch.Tensor): A square image to unproject (S, S, 3).
        mask (torch.Tensor): A binary mask for the image (S, S).
        depth (torch.Tensor): The depth map of the image (S, S).
        camera: The Pytorch3D camera to render the image.
    
    Returns:
        points (torch.Tensor): The 3D points of the unprojected image (N, 3).
        rgba (torch.Tensor): The rgba color values corresponding to the unprojected
            points (N, 4).
    """
    device = camera.device
    assert image.shape[0] == image.shape[1], "Image must be square."
    image_shape = image.shape[0]
    ndc_pixel_coordinates = torch.linspace(1, -1, image_shape)
    Y, X = torch.meshgrid(ndc_pixel_coordinates, ndc_pixel_coordinates,indexing='ij') # i have added indesing to get rid of warning
    xy_depth = torch.dstack([X, Y, depth])
    points = camera.unproject_points(
        xy_depth.to(device), in_ndc=False, from_ndc=False, world_coordinates=True,
    )
    points = points[mask > 0.5]
    rgb = image[mask > 0.5]
    rgb = rgb.to(device)

    # For some reason, the Pytorch3D compositor does not apply a background color
    # unless the pointcloud is RGBA.
    alpha = torch.ones_like(rgb)[..., :1]
    rgb = torch.cat([rgb, alpha], dim=1)

    return points, rgb

def render_save_360_gif(obj_data, # point_cloud or a mesh
                        object_type="point_cloud", # "point_cloud" or "mesh"
                        num_views=120,
                        distance=3, elevation=0, # for R, T computation
                        image_size=256,
                        gif_save_path='random.gif',
                        loop=0,
                        device=None):
    
    if device is None:
        device = get_device()  

    azimuths = torch.linspace(0, 360, num_views).tolist()

    images = []
    for azimuth in azimuths:
        R, T = pytorch3d.renderer.look_at_view_transform(distance, elevation, azimuth, device=device)  
        
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        if object_type == "point_cloud":
            renderer = get_points_renderer(image_size=image_size, device=device)
        elif object_type == "mesh":
            renderer = get_mesh_renderer(image_size=image_size, device=device)
        else:   
            raise ValueError("object_type should be either 'point_cloud' or 'mesh'")
        
        rend = renderer(obj_data, cameras=cameras, lights=lights)
        rend = (rend.detach().cpu().numpy()[0, ..., :3] * 255).astype('uint8')
        images.append(rend)
    
    duration = 1000 // 15
    imageio.mimsave(gif_save_path, images, duration=duration, loop=loop)


# vox renderer
def render_voxels(voxel_, image_size=256, dist=3, elev=10, azim=180, save_dir=None, save_name='random.gif'):
    device = get_device()

    vox_mesh = pytorch3d.ops.cubify(voxel_, thresh=0.5, device=device)
    
    vertices = vox_mesh.verts_list()[0]
    faces = vox_mesh.faces_list()[0]
    
    # option1 for texture
    # color = torch.tensor([0.7, 0.7, 1], device=device)
    color = torch.tensor([0.2, 0.5, 0.8], device=device)
    textures = torch.ones_like(vertices, device=device)
    textures = textures * color
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0))

    # option2 for texture
    # textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    # textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )

    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)

    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=elev, azim=azim)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)

    render_save_360_gif(mesh, object_type="mesh", gif_save_path=f'{save_dir}/{save_name}')


# point cloud renderer
def render_ptcloud(pt_cloud_data, dist=3, save_dir=None, save_name='random.gif'):
    device = get_device()
    if isinstance(pt_cloud_data, torch.Tensor):
        B, N, _ = pt_cloud_data.shape
        color = (pt_cloud_data - pt_cloud_data.min()) / (pt_cloud_data.max() - pt_cloud_data.min())
        pt_cloud_data_ = Pointclouds(points=pt_cloud_data, features=color)
    render_save_360_gif(pt_cloud_data_, object_type="point_cloud", image_size=512, distance=dist, gif_save_path=f'{save_dir}/{save_name}')


def render_mesh(mesh, image_size=256, dist=2, elev=10, azim=180, save_dir=None, save_name='random.gif'):
    device = get_device()
    mesh = mesh.to(device)

    vertices = mesh.verts_list()[0]
    color = torch.tensor([0.2, 0.5, 0.8], device=device)
    textures = torch.ones_like(vertices, device=device)
    textures = textures * color
    mesh.textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0))


    render_save_360_gif(mesh, object_type="mesh", num_views=120, distance=dist, 
                          elevation=elev, image_size=image_size,
                          gif_save_path=f'{save_dir}/{save_name}', device=device)