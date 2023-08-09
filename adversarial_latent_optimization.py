import pdb
import os
import argparse
from get_model import get_model
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm, trange
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
import time
import torchvision
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils_sgm import register_hook_for_resnet, register_hook_for_densenet


'''
CUDA_VISIBLE_DEVICES=0 python3 adversarial_latent_optimization.py --model mnv2 --beta 0.1 --alpha 0.04 --steps 10 --norm 2 --start 0 --end 1000 --mu 1 --eps 0.1 
'''
############## Initialize #####################
parser = argparse.ArgumentParser(description='Adversarial Content Attack')
parser.add_argument('--model', type=str, default='resnet50', help='model')
parser.add_argument('--alpha', type=float, default=0.04, help='step size')
parser.add_argument('--beta', type=float, default=0.1, help='mse factor')
parser.add_argument('--eps', type=float, default=0.1, help='perturbation value')
parser.add_argument('--steps', type=int, default=10, help='attack steps')
parser.add_argument('--norm', type=int, default=2, help='loss norm')
parser.add_argument('--lp', type=str, default='linf', help='perturbation norm')
parser.add_argument('--start', default=0, type=int, help='img start')
parser.add_argument('--end', default=1000, type=int, help='img end')
parser.add_argument('--prefix', type=str, default='ACA-test', help='filename')
parser.add_argument('--target', default=-1, type=int, help='target class, -1 is untargeted attack')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--mu', default=1, type=float, help='momentum factor')


args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
torch.Generator().manual_seed(args.seed)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==> Preparing Model..')
image_size = (224, 224)
if args.model == 'vit' or args.model == 'adv_resnet152_denoise':
    print('Using 0.5 Nor...')
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
elif args.model == 'mvit':
    mean = [0, 0, 0]
    std = [1, 1, 1] 
    image_size = (320, 320)
else:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


mean = torch.Tensor(mean).cuda()
std = torch.Tensor(std).cuda()

net = get_model(args.model)
if device == 'cuda':
    net.to(device)
    cudnn.benchmark = True
net.eval()
# # Apply SGM
# if args.model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
#     register_hook_for_resnet(net, arch=args.model, gamma=0.5)
# else:
#     raise ValueError('Current code only supports resnet/densenet. '
#                         'You can extend this code to other architectures.')
net.cuda()

class EmptyControl:
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

@torch.no_grad()
def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents

def limitation01(y):
    idx = (y > 1)
    y[idx] = (torch.tanh(1000*(y[idx]-1))+10000)/10001
    idx = (y < 0)
    y[idx] = (torch.tanh(1000*(y[idx])))/10000
    return y

def norm_l2(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

@torch.no_grad()
def adversarial_latent_optimization(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    label=None,
    raw_img=None
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    print("Latent", latent.shape, "Latents", latents.shape)
    model.scheduler.set_timesteps(num_inference_steps)

    best_latent = latents
    ori_latents = latents.clone().detach()
    adv_latents = latents.clone().detach()
    print(latents.max(), latents.min())
    success = True
    momentum = 0
    for k in range(args.steps):
        latents = adv_latents
        for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
            # print(i, t)
            if uncond_embeddings_ is None:
                context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            else:
                context = torch.cat([uncond_embeddings_, text_embeddings])
            latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)

        image = None
        with torch.enable_grad():
            latents_last = latents.detach().clone()
            latents_last.requires_grad = True
            latents_t = (1 / 0.18215 * latents_last)
            image = model.vae.decode(latents_t)['sample']
            image = (image / 2 + 0.5)

            # print(4, image.max(), image.min())
            image = limitation01(image)
            image_m = F.interpolate(image, image_size)

            # print(1, image_m.max(), image_m.min())
            image_m = image_m - mean[None,:,None,None]
            image_m = image_m / std[None,:,None,None]
            outputs = net(image_m)
            _, predicted = outputs.max(1)

            if args.target == -1:
                if label != predicted:
                    best_latent = adv_latents
                    success = False
            else:
                if args.target == predicted:
                    best_latent = adv_latents
                    success = False


            if args.target == -1:
                loss_ce = torch.nn.CrossEntropyLoss()(outputs, torch.Tensor([label]).long().cuda())
            else:
                loss_ce = -torch.nn.CrossEntropyLoss()(outputs, torch.Tensor([args.target]).long().cuda())
                
            # print(3, image_m.max(), image_m.min(), raw_img.max(), raw_img.min())
            loss_mse = args.beta * torch.norm(image_m-raw_img, p=args.norm).mean()
            loss = loss_ce - loss_mse
            loss.backward()

            print('*' * 50)
            print('Loss', loss.item(), 'Loss_ce', loss_ce.item(), 'Loss_mse', loss_mse.item())
            print(k, 'Predicted:', label, predicted, loss.item())
            # print('Grad:', latents_last.grad.min(), latents_last.grad.max())

        l1_grad = latents_last.grad / torch.norm(latents_last.grad, p=1)
        # print('L1 Grad:', l1_grad.min(), l1_grad.max())
        momentum = args.mu * momentum + l1_grad
        if args.lp == 'linf':
            adv_latents = adv_latents + torch.sign(momentum) * args.alpha
            noise = (adv_latents - ori_latents).clamp(-args.eps, args.eps)
        elif args.lp == 'l2':
            adv_latents = adv_latents + args.alpha * momentum.detach() / norm_l2(momentum.detach())
            noise = (adv_latents - ori_latents) * args.eps / norm_l2(adv_latents - ori_latents).clamp(min=args.eps)
        adv_latents = ori_latents + noise
        latents = adv_latents.detach()

    if success:
        best_latent = latents

    # Return Best Attack
    latents = best_latent
    for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
        # print(i, t)
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
    latents = (1 / 0.18215 * latents)
    image = model.vae.decode(latents)['sample']
    image = (image / 2 + 0.5)
    # print(4, image.max(), image.min())
    image = limitation01(image)
    image = F.interpolate(image, image_size)
    # print(2, image.max(), image.min())

    image = image.clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image, best_latent, success


if __name__ == '__main__':
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    MY_TOKEN = 'your_token'
    LOW_RESOURCE = False 
    NUM_DDIM_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler).to(device)
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm_stable.tokenizer
    
    image_nums = 1000
    img_path = 'temp/1000/inversion'
    raw_img_path = 'third_party/Natural-Color-Fool/dataset/images'
    all_prompts = open('temp/1000/prompts.txt').readlines()
    all_latents = torch.load('temp/1000/all_latents.pth')
    all_uncons = torch.load('temp/1000/all_uncons.pth')
    all_labels = open('third_party/Natural-Color-Fool/dataset/labels.txt').readlines()

    img_list = os.listdir(img_path)
    img_list.sort()
    cnt = 0
    save_path = './temp/' + args.prefix + '-' + str(args.mu) + '-' + args.model + '-' + str(args.alpha) + '-' + str(args.beta) + '-' + str(args.norm) + '-' + str(args.steps) + '-Clip-' + str(args.eps) + '-' + args.lp + '-' + str(args.target) + '/'
    print("Save Path:", save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in trange(args.start, args.end):
        if not os.path.exists(os.path.join(save_path, img_list[i].split('.')[0]+'.png')):
            img_path = os.path.join(img_path, img_list[i])
            idx = int(img_list[i].split('.')[0]) 
            prompt = all_prompts[idx].strip()
            x_t = all_latents[idx].cuda()
            uncond_embeddings = all_uncons[idx].cuda()
            label = int(all_labels[idx].strip()) - 1
            print(idx, label)
            pil_image = Image.open(os.path.join(raw_img_path, str(idx+1)+'.png')).convert('RGB').resize(image_size)
            raw_img = (torch.tensor(np.array(pil_image), device=device).unsqueeze(0)/255.).permute(0, 3, 1, 2)
            raw_img = raw_img - mean[None,:,None,None]
            raw_img = raw_img / std[None,:,None,None]

            prompts = [prompt]
            controller = EmptyControl()
            image_inv, x_t, success = adversarial_latent_optimization(ldm_stable, prompts, controller, latent=x_t, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=None, uncond_embeddings=uncond_embeddings, label=label, raw_img=raw_img)
            ptp_utils.view_images([image_inv[0]], prefix=os.path.join(save_path, img_list[i].split('.')[0]))
            cnt += success
            print("Acc: ", cnt, '/', (i-args.start+1))
        
        else:
            print(os.path.join(save_path, img_list[i].split('.')[0]), " has existed!")


