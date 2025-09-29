#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.polarization_utils import stokes_fac_from_normal
from utils.sh_utils import eval_sh
from utils.general_utils import get_env_rayd1, get_env_rayd2

def render_env_map(pc: GaussianModel):
    env_cood1 = sample_cubemap_color(get_env_rayd1(512,1024), pc.get_envmap)
    env_cood2 = sample_cubemap_color(get_env_rayd2(512,1024), pc.get_envmap)
    return {'env_cood1': env_cood1, 'env_cood2': env_cood2}

pixel_camera = None
def sample_camera_rays(HWK, R, T):
    H,W,K = HWK
    R = R.T # NOTE!!! the R rot matrix is transposed save in 3DGS
    global pixel_camera
    if pixel_camera is None or pixel_camera.shape[0] != H:
        K = K.astype(np.float32)
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32),
                        indexing='xy')
        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_camera = torch.tensor(pixel_camera).cuda()
    rays_o = (-R.T @ T.unsqueeze(-1)).flatten()
    pixel_world = (pixel_camera - T[None, None]).reshape(-1, 3) @ R
    rays_d = pixel_world - rays_o[None]
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    rays_d = rays_d.reshape(H,W,3)
    return rays_d
def reflection(rayd, normal):
    refl = rayd - 2*normal*torch.sum(rayd*normal, dim=-1, keepdim=True)
    return refl
def sample_cubemap_color(rays_d, env_map):
    H,W = rays_d.shape[:2]
    outcolor = torch.sigmoid(env_map(rays_d.reshape(-1,3)))
    outcolor = outcolor.reshape(H,W,3).permute(2,0,1)
    return outcolor
def get_refl_color(envmap: torch.Tensor, HWK, R, T, normal_map): #RT W2C
    rays_d = sample_camera_rays(HWK, R, T)
    rays_d = reflection(rays_d, normal_map)
    #rays_d = rays_d.clamp(-1, 1) # avoid numerical error when arccos
    return sample_cubemap_color(rays_d, envmap)


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, patch_size: list, scaling_modifier = 1.0, override_color = None, initial_stage= False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    viewpoint_camera.to_device()
    viewpoint_camera.update()

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        patch_bbox=viewpoint_camera.random_patch(patch_size[0], patch_size[1]),
        prcppoint=viewpoint_camera.prcppoint,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        config=pc.config
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, rendered_normal, rendered_depth, rendered_opac,  radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    if initial_stage:
        return {"render": rendered_image, "normal": rendered_normal, "depth": rendered_depth, "opac": rendered_opac,
                    "viewspace_points": screenspace_points, "visibility_filter" : radii > 1, "radii": radii}
    rendered_normal_world = torch.matmul(viewpoint_camera.world_view_transform[:3,:3][None,None,...], rendered_normal.permute(1,2,0)[...,None]).squeeze()
    refl_color = get_refl_color(pc.get_envmap, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, rendered_normal_world)
    rays_o = -viewpoint_camera.R @ viewpoint_camera.T.unsqueeze(-1)
    rays_o = rays_o.flatten()
    stokes_diff, stokes_spec, _ = stokes_fac_from_normal(rays_o[...,None],viewpoint_camera.view_direction[...,None,:],
                                                                        rendered_normal_world[...,None,:],
                                                                        ret_spec=True,
                                                                        clip_spec=True)
    
    stokes_diff_with_grad = rendered_image.permute(1,2,0)[...,None] * stokes_diff.squeeze()[...,None,:]
    stokes_spec_with_grad = refl_color.permute(1,2,0)[...,None] * stokes_spec.squeeze()[...,None,:]
    rendered_image_with_ref = stokes_diff_with_grad[...,0].permute(2,0,1) +  stokes_spec_with_grad[...,0].permute(2,0,1)
    
    stokes_diff_no_grad = rendered_image.permute(1,2,0)[...,None].detach() * stokes_diff.squeeze()[...,None,:]
    stokes_spec_no_grad = refl_color.permute(1,2,0)[...,None].detach() * stokes_spec.squeeze()[...,None,:]
    rendered_s1 = stokes_diff_no_grad[...,1].permute(2,0,1) +  stokes_spec_no_grad[...,1].permute(2,0,1)
    rendered_s2 = stokes_diff_no_grad[...,2].permute(2,0,1) +  stokes_spec_no_grad[...,2].permute(2,0,1)
    
    return {"render": rendered_image, "normal": rendered_normal, "depth": rendered_depth, "opac": rendered_opac,
            "rendered_normal_world": rendered_normal_world,"rendered_s1":rendered_s1, "rendered_s2":rendered_s2,
            "viewspace_points": screenspace_points, "visibility_filter" : radii > 1, "radii": radii, 
            "refl_color": refl_color, "rendered_image_with_ref":rendered_image_with_ref}