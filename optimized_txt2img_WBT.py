import sys
import cgitb
import json
import argparse, os, re
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger
from transformers import logging
# from samplers import CompVisDenoiser
logging.set_verbosity_error()

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


config = "optimizedSD/v1-inference.yaml"
DEFAULT_CKPT = "models/ldm/stable-diffusion-v1/model.ckpt"

# Opening JSON file
with open('C:/TempJson/sample.json', 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

#get elements from json
prompt = json_object.get('prompt')
outdir = json_object.get('outdir')
ddim_steps = json_object.get('ddim_steps')
n_iter = json_object.get("n_iter")
H = json_object.get('h')
W = json_object.get('w')
C = json_object.get('c')
f = json_object.get('f')
n_samples = json_object.get('num_images')
scale = json_object.get('scale')
seed = json_object.get('seed')
precision = json_object.get('precision')
format = json_object.get('format')
sampler = json_object.get('sampler')

#add default values
ckpt = DEFAULT_CKPT
unet_bs = 1
device = "cuda"
turbo = True
fixed_code = True
sn_rows = 0
from_file = ""
ddim_eta = 0.0

#opt = parser.parse_args()

tic = time.time()
os.makedirs(outdir, exist_ok=True)
outpath = outdir
grid_count = len(os.listdir(outpath)) - 1

if seed == None:
    seed = randint(0, 1000000)
seed_everything(seed)

# Logging
logger(json_object, log_csv = "logs/txt2img_logs.csv")

sd = load_model_from_config(f"{ckpt}")
li, lo = [], []
for key, value in sd.items():
    sp = key.split(".")
    if (sp[0]) == "model":
        if "input_blocks" in sp:
            li.append(key)
        elif "middle_block" in sp:
            li.append(key)
        elif "time_embed" in sp:
            li.append(key)
        else:
            lo.append(key)
for key in li:
    sd["model1." + key[6:]] = sd.pop(key)
for key in lo:
    sd["model2." + key[6:]] = sd.pop(key)

config = OmegaConf.load(f"{config}")

model = instantiate_from_config(config.modelUNet)
_, _ = model.load_state_dict(sd, strict=False)
model.eval()
model.unet_bs = unet_bs
model.cdevice = device
model.turbo = turbo

modelCS = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()
modelCS.cond_stage_model.device = device

modelFS = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)
modelFS.eval()
del sd

if device != "cpu" and precision == "autocast":
    model.half()
    modelCS.half()

start_code = None
if fixed_code:
    start_code = torch.randn([n_samples, C, H // f, W //f], device=device)


batch_size = n_samples
n_rows = sn_rows if sn_rows > 0 else batch_size
if not from_file:
    assert prompt is not None
    prompt = prompt
    print(f"Using prompt: {prompt}")
    data = [batch_size * [prompt]]

else:
    print(f"reading prompts from {from_file}")
    with open(from_file, "r") as f:
        text = f.read()
        print(f"Using prompt: {text.strip()}")
        data = text.splitlines()
        data = batch_size * list(data)
        data = list(chunk(sorted(data), batch_size))


if precision == "autocast" and device != "cpu":
    precision_scope = autocast
else:
    precision_scope = nullcontext

seeds = ""
with torch.no_grad():

    all_samples = list()
    for n in trange(n_iter, desc="Sampling"):
        for prompts in tqdm(data, desc="data"):

            sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompts[0])))[:150]
            os.makedirs(sample_path, exist_ok=True)
            base_count = len(os.listdir(sample_path))

            with precision_scope("cuda"):
                modelCS.to(device)
                uc = None
                if scale != 1.0:
                    uc = modelCS.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)

                subprompts, weights = split_weighted_subprompts(prompts[0])
                if len(subprompts) > 1:
                    c = torch.zeros_like(uc)
                    totalWeight = sum(weights)
                    # normalize each "sub prompt" and add it
                    for i in range(len(subprompts)):
                        weight = weights[i]
                        # if not skip_normalize:
                        weight = weight / totalWeight
                        c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                else:
                    c = modelCS.get_learned_conditioning(prompts)

                shape = [n_samples, C, H // f, W // f]

                if device != "cpu":
                    mem = torch.cuda.memory_allocated() / 1e6
                    modelCS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)

                samples_ddim = model.sample(
                    S=ddim_steps,
                    conditioning=c,
                    seed=seed,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    eta=ddim_eta,
                    x_T=start_code,
                    sampler = sampler,
                )

                modelFS.to(device)

                print(samples_ddim.shape)
                print("saving images")
                for i in range(batch_size):

                    x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                    x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                    Image.fromarray(x_sample.astype(np.uint8)).save(
                        os.path.join(sample_path, "seed_" + str(seed) + "_" + f"{base_count:05}.{format}")
                    )
                    seeds += str(seed) + ","
                    seed += 1
                    base_count += 1

                if device != "cpu":
                    mem = torch.cuda.memory_allocated() / 1e6
                    modelFS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)
                del samples_ddim
                print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

toc = time.time()

time_taken = (toc - tic) / 60.0

print(
    (
        "Samples finished in {0:.2f} minutes and exported to "
        + sample_path
        + "\n Seeds used = "
        + seeds[:-1]
    ).format(time_taken)
)
