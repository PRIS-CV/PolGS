import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_env_map
import torchvision
from utils.general_utils import safe_state, poisson_mesh
from utils.image_utils import  depth2rgb, normal2rgb, depth2normal, match_depth, resample_points, grid_prune
from argparse import ArgumentParser
from torchvision.utils import save_image
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2


def render_set(model_path, use_mask, name, iteration, views, gaussians, pipeline, background, write_image, poisson_depth):
    render_path = os.path.join(model_path, name, "{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "{}".format(iteration), "gt")
    info_path = os.path.join(model_path, name, "{}".format(iteration), "info")
    normal_path = os.path.join(model_path, name, "{}".format(iteration), "normal")
    
    speculardirs = os.path.join(model_path, name, "{}".format(iteration), "specular")
    diffuse_dirs = os.path.join(model_path, name, "{}".format(iteration), "diffuse")
    depth_dirs = os.path.join(model_path, name, "{}".format(iteration), "depth")
   
    makedirs(speculardirs, exist_ok=True)
    makedirs(diffuse_dirs, exist_ok=True)
    makedirs(depth_dirs, exist_ok=True)
    ########################
    

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(info_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)

    ltres = render_env_map(gaussians)

    torchvision.utils.save_image(ltres['env_cood1']**(1/2.2), os.path.join(model_path, name, "{}".format(iteration), 'light1.png'))
    torchvision.utils.save_image(ltres['env_cood2']**(1/2.2), os.path.join(model_path, name, "{}".format(iteration), 'light2.png'))

    if name == 'train':
        bound = None
        occ_grid, grid_shift, grid_scale, grid_dim = gaussians.to_occ_grid(0.0, 512, bound)

    resampled = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        background = torch.zeros((3), dtype=torch.float32, device="cuda")
        render_pkg = render(view, gaussians, pipeline, background, [float('inf'), float('inf')])

        image, normal, depth, opac, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["opac"], \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        reflec_image = render_pkg["refl_color"]
        image = image
        final_image = image + reflec_image

        mask_gt = view.get_gtMask(use_mask)
        gt_s0,_,_ = view.get_gtStokes(background, use_mask)
        mask_vis = (opac.detach() > 1e-5)
        depth_range = [0, 20]
        mask_clip = (depth > depth_range[0]) * (depth < depth_range[1])

        normal = torch.nn.functional.normalize(normal, dim=0) * mask_vis
        if view.HWK[0] == 512 and view.HWK[1] == 512:
            normal = torch.cat([-normal[:1], -normal[1:2], normal[2:]])
        d2n = depth2normal(depth, mask_vis, view)

        if name == 'train':
            pts = resample_points(view, depth, normal, image, mask_vis * mask_gt * mask_clip)
            grid_mask = grid_prune(occ_grid, grid_shift, grid_scale, grid_dim, pts[..., :3], thrsh=1)
            clean_mask = grid_mask #* mask_mask
            pts = pts[clean_mask]
            resampled.append(pts.cpu())

        if write_image:
            normal_wrt = normal2rgb(normal, mask_vis)
            depth_wrt = depth2rgb(depth, mask_vis)
            d2n_wrt = normal2rgb(d2n, mask_vis)
            normal_wrt += background[:, None, None] * (~mask_vis).expand_as(image) * mask_gt
            depth_wrt += background [:, None, None]* (~mask_vis).expand_as(image) * mask_gt
            d2n_wrt += background[:, None, None] * (~mask_vis).expand_as(image) * mask_gt
            outofmask = mask_vis * (1 - mask_gt)
            mask_vis_wrt = outofmask * (opac - 1) + mask_vis
            img_wrt = torch.cat([gt_s0.cuda(), final_image, image, reflec_image, normal_wrt, depth_wrt], 2)
            wrt_mask = torch.cat([mask_gt, mask_vis_wrt, mask_vis_wrt, mask_vis_wrt, mask_vis_wrt, mask_vis_wrt], 2)
            img_wrt = torch.cat([img_wrt, wrt_mask], 0)
            save_image(img_wrt.cpu(), os.path.join(info_path, '{}'.format(view.image_name) + f".png"))
            save_image(final_image.cpu(), os.path.join(render_path, '{}'.format(view.image_name) + ".png"))
            save_image((torch.cat([gt_s0.cuda(), mask_gt], 0)).cpu(), os.path.join(gts_path, '{}'.format(view.image_name) + ".png"))
            normal_wrt = normal_wrt * mask_gt
            cv2.imwrite(os.path.join(normal_path, '{0:05d}.png'.format(idx)), (normal_wrt.permute(1,2,0).cpu().numpy()[...,::-1]*65535).astype(np.uint16))
            save_image((torch.cat([reflec_image, mask_gt], 0)).cpu(), os.path.join(speculardirs, '{0:05d}.png'.format(idx)))
            save_image(image[None,...]*mask_gt, os.path.join(diffuse_dirs, '{0:05d}.png'.format(idx)))
            save_image(depth_wrt[None,...]*mask_gt, os.path.join(depth_dirs, '{0:05d}.png'.format(idx)))
            
        view.to_cpu()

    if name == 'train':
        resampled = torch.cat(resampled, 0)
        mesh_path = f'{model_path}/poisson_mesh_{poisson_depth}'
        poisson_mesh(mesh_path, resampled[:, :3], resampled[:, 3:6], resampled[:, 6:], poisson_depth, 3 * 1e-5)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, write_image: bool, poisson_depth: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset)
        scales = [1]
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=scales)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not skip_train:
             render_set(dataset.model_path, True, "train", scene.loaded_iter, scene.getTrainCameras(scales[0]), gaussians, pipeline, background, write_image, poisson_depth)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--img", action="store_true")
    parser.add_argument("--depth", default=8, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.img, args.depth)