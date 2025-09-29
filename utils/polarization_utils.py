import numpy as np



def normalize(v):
    import torch
    return torch.nn.functional.normalize(v, dim=-1)

def normalize_np(v):
    return  v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),
                     1e-12)

def stokes_fac_from_normal(rays_o, rays_d, normal, 
                           train_mode=False,
                           ret_spec=False,
                           clip_spec=False):
    import torch

    # Add singleton dimension for Num_lights
    rays_d = rays_d[..., None, :]
    normal = normal[..., None, :]

    cos = torch.cos 
    sin = torch.sin 
    acos = torch.acos
    atan2 = torch.atan2
    sqrt = torch.sqrt
    normize = normalize
    stack = torch.stack
    acos = lambda x: torch.acos(torch.clamp(x, min=-1.+1e-7, max=1.-1e-7))
    dot = lambda x, y: (x * y).sum(-1, keepdim=True)
    clamp = lambda x, y: torch.clamp(x, min=y)
    mask_fn = lambda x, mask: torch.where(mask, x, torch.zeros_like(x))
    cross = lambda x, y: torch.cross(x, y.broadcast_to(x.shape), dim=-1)

    # Helper variables    
    eta = 1.5
    n = normal
    o = -rays_d
    # h = n 
    dotno = dot(n, o)
    n_o = normize(n - dotno * o)
    # h_o = n_o

    x_o = normize(stack([-o[..., 1], o[..., 0], torch.zeros_like(o[..., 2])], -1))
    y_o = cross(x_o, o)
    phi_o = atan2(-dot(n_o, x_o), dot(n_o, y_o))
    psi_o = phi_o

    eta_i_1, eta_i_2 = 1.0, eta
    theta_i_1 = acos(dotno)
    theta_i_2 = acos(sqrt(clamp(1 - (sin(theta_i_1) / eta) ** 2, 1e-7)))

    eta_o_1, eta_o_2 = eta, 1.0
    theta_o_2 = acos(dotno)
    theta_o_1 = acos(sqrt(clamp(1 - (sin(theta_o_2) / eta) ** 2, 1e-7)))

    theta_d = acos(dotno)
    eta_r_1, eta_r_2 = 1.0, eta
    theta_r_1 = theta_d
    theta_r_2 = acos(sqrt(clamp(1 - (sin(theta_r_1) / eta) ** 2, 1e-7)))

    T_i__perp = (2 * eta_i_1 * cos(theta_i_1)) ** 2 / clamp((eta_i_1 * cos(theta_i_1) + eta_i_2 * cos(theta_i_2)) ** 2, 1e-7)
    T_i__perp *= (cos(theta_i_1) > 1e-7)
    T_i__para = (2 * eta_i_1 * cos(theta_i_1)) ** 2 / clamp((eta_i_1 * cos(theta_i_2) + eta_i_2 * cos(theta_i_1)) ** 2, 1e-7)
    T_i__para *= (cos(theta_i_1) > 1e-7)

    T_o__perp = (2 * eta_o_1 * cos(theta_o_1)) ** 2 / clamp((eta_o_1 * cos(theta_o_1) + eta_o_2 * cos(theta_o_2)) ** 2, 1e-7)
    T_o__para = (2 * eta_o_1 * cos(theta_o_1)) ** 2 / clamp((eta_o_1 * cos(theta_o_2) + eta_o_2 * cos(theta_o_1)) ** 2, 1e-7)
    T_o__plus, T_o__min = 0.5 * (T_o__perp + T_o__para), 0.5 * (T_o__perp - T_o__para)

    R__perp = (eta_r_1 * cos(theta_r_1) - eta_r_2 * cos(theta_r_2)) ** 2 / clamp((eta_r_1 * cos(theta_r_1) + eta_r_2 * cos(theta_r_2)) ** 2, 1e-7)
    R__para = (eta_r_1 * cos(theta_r_2) - eta_r_2 * cos(theta_r_1)) ** 2 / clamp((eta_r_1 * cos(theta_r_2) + eta_r_2 * cos(theta_r_1)) ** 2, 1e-7)
    R__plus, R__min = 0.5 * (R__perp + R__para), 0.5 * (R__perp - R__para)

    stokes_diff_fac = stack([torch.ones_like(T_o__min), T_o__min / T_o__plus * cos(2 * phi_o), -T_o__min / T_o__plus * sin(2 * phi_o)], -1)
    stokes_spec_fac = stack([torch.ones_like(R__plus), R__min / R__plus * cos(2 * psi_o), -R__min / R__plus * sin(2 * psi_o)], -1)

    if clip_spec:
        spec_mask = dotno > 1e-7
        stokes_spec_fac = mask_fn(stokes_spec_fac, spec_mask[..., None])
        R__plus = mask_fn(R__plus, spec_mask)

    return stokes_diff_fac, stokes_spec_fac, R__plus[..., None]