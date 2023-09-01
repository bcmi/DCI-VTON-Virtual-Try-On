import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision

from ldm.data.cp_dataset import CPDataset
from ldm.resizer import Resizer
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.deepfashions import DFPairDataset

import clip
from torchvision.transforms import Resize


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="which gpu to use",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=30,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_imgs",
        type=int,
        default=100,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given reference image. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--unpaired",
        action='store_true',
        help="if enabled, uses the same starting code across samples "
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        help="path to dataroot of the dataset",
        default=""
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda:{}".format(opt.gpu_id)) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device)

    config = OmegaConf.load(f"{opt.config}")
    version = opt.config.split('/')[-1].split('.')[0]
    model = load_model_from_config(config, f"{opt.ckpt}")

    # model = model.to(device)
    dataset = CPDataset(opt.dataroot, opt.H, mode='test', unpaired=opt.unpaired)
    loader = DataLoader(dataset, batch_size=opt.n_samples, shuffle=False, num_workers=4, pin_memory=True)
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    result_path = os.path.join(outpath, "result")
    os.makedirs(result_path, exist_ok=True)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    iterator = tqdm(loader, desc='Test Dataset', total=len(loader))
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for data in iterator:
                    mask_tensor = data['inpaint_mask']
                    inpaint_image = data['inpaint_image']
                    ref_tensor = data['ref_imgs']
                    feat_tensor = data['warp_feat']
                    image_tensor = data['GT']
                    # filename = data['file_name']

                    test_model_kwargs = {}
                    test_model_kwargs['inpaint_mask'] = mask_tensor.to(device)
                    test_model_kwargs['inpaint_image'] = inpaint_image.to(device)
                    feat_tensor = feat_tensor.to(device)
                    ref_tensor = ref_tensor.to(device)

                    uc = None
                    if opt.scale != 1.0:
                        uc = model.learnable_vector
                        uc = uc.repeat(ref_tensor.size(0), 1, 1)
                    c = model.get_learned_conditioning(ref_tensor.to(torch.float16))
                    c = model.proj_out(c)

                    # z_gt = model.encode_first_stage(image_tensor.to(device))
                    # z_gt = model.get_first_stage_encoding(z_gt).detach()

                    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image'] = z_inpaint
                    test_model_kwargs['inpaint_mask'] = Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(
                        test_model_kwargs['inpaint_mask'])

                    warp_feat = model.encode_first_stage(feat_tensor)
                    warp_feat = model.get_first_stage_encoding(warp_feat).detach()

                    ts = torch.full((1,), 999, device=device, dtype=torch.long)
                    start_code = model.q_sample(warp_feat, ts)

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_code,
                                                     test_model_kwargs=test_model_kwargs)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_sample_result = x_samples_ddim
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image = x_samples_ddim
                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                    x_source = torch.clamp((image_tensor + 1.0) / 2.0, min=0.0, max=1.0)
                    x_result = x_checked_image_torch * (1 - mask_tensor) + mask_tensor * x_source

                    resize = transforms.Resize((opt.H, int(opt.H / 256 * 192)))

                    if not opt.skip_save:

                        def un_norm(x):
                            return (x + 1.0) / 2.0

                        for i, x_sample in enumerate(x_result):
                            filename = data['file_name'][i]
                            # filename = data['file_name']
                            save_x = resize(x_sample)
                            save_x = 255. * rearrange(save_x.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(save_x.astype(np.uint8))
                            img.save(os.path.join(result_path, filename[:-4] + ".png"))

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
