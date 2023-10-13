import torch
import numpy as np
import torch.nn.functional as F

def conditional_gen(model, pC=None):
    # select whether generate at random or not
    if pC is None:
        pC = 5 * torch.randn((8, model.n_images, model.encoder.c_dim), device=model.device)
        # pC = torch.softmax(pC, dim=-1)

    zs = torch.randn((8, model.n_images, model.encoder.latent_dim), device=model.device)

    latents = []   
    for _ in range(model.n_images):
        for i in range(len(model.c_split)):
            latents.append(zs[:,i,:])
            latents.append(F.gumbel_softmax(pC[:, i, :], tau=1, hard=True, dim=-1)) 
    
    # generated images
    decode = model.decoder(torch.cat(latents, dim=1)).detach()

    return decode

def recon_visaulization(out_dict):
    images = out_dict['INPUTS'].detach()[:8]
    recons = out_dict['RECS'].detach()[:8]
    return torch.cat([images, recons], dim=0 )
