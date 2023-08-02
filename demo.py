import os
import time
from PIL import Image
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from templates import *
from iterative_projected_gradient_fast import PGDAttack
from attack_utils import load_test_models, Cos_Loss, read_img, Net_fast, OhemCELoss
from torchvision.utils import save_image
import glob
import cv2
import numpy as np

from external.face_makeup.model import BiSeNet
from external.face_makeup.test import vis_parsing_maps


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

def vis_parsing_maps(im, parsing_anno, stride=1, save_im=True, save_path='parsing_map_on_im.png'):
    # Colors for all 20 parts
    part_colors = [[0, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(0, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    # pi = 0
    # index = np.where(vis_parsing_anno == pi)
    # vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    # vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0., vis_parsing_anno_color, 1., 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path, vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return vis_parsing_anno

def generate(args):
    eps = args.eps
    iter = args.iter
    T_enc =  args.T_enc
    T_atk = args.T_atk
    T_inf = args.T_inf
    start_t = args.start_t
    attack_iter = args.attack_iter
    attack_inf_iter = args.attack_inf_iter
    repeat_times = args.repeat_times
    cnt_skip = args.cnt_skip
    lam = args.lam
    vis_full = args.vis_full

    # set up FR models
    model_names = args.model_names # ['ir152', 'irse50', 'facenet']
    test_models = load_test_models(model_names)

    # set up diffae model
    conf = ffhq256_autoenc()
    model = LitModel(conf)
    state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.cuda()

    # set up face parsing model
    if lam > 0:
        net_parse = BiSeNet(n_classes=19)
        net_parse.load_state_dict(torch.load('./external/face_makeup/cp/79999_iter.pth'))
        net_parse.cuda()
        net_parse.eval()
        loss_parse_fn = OhemCELoss(thresh=0.7, n_min=256 * 256 // 16, ignore_lb=0)
    else:
        loss_parse_fn = None
        net_parse = None

    # set up attacker
    net = Net_fast(test_models, model).cuda()
    test_attacker = PGDAttack(predict=net,
                              loss_fn=Cos_Loss,
                              eps=eps,
                              eps_iter=2 * eps / (iter * attack_iter * repeat_times),
                              nb_iter=iter,
                              T_atk=T_atk,
                              T_enc=T_enc,
                              start_t=start_t,
                              attack_iter=attack_iter,
                              attack_inf_iter=attack_inf_iter,
                              repeat_times=repeat_times,
                              cnt_skip=cnt_skip,
                              lam=lam,
                              clip_min=-1e6,
                              clip_max=1e6,
                              targeted=True,
                              rand_init=False,
                              loss_parse_fn=loss_parse_fn,
                              parse=net_parse,
                              vis_full = args.vis_full)

    # Target embedding
    target = read_img(args.target_path, 0., 1., 'cuda')
    with torch.no_grad():
        target_embeding, _, _ = net(target)

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    source_paths = os.listdir(args.source_dir)
    timer = Timer()
    time_list = []
    for source_path in tqdm(source_paths):
        source_name = source_path.replace('.jpg', '.png')
        source_path = os.path.join(args.source_dir, source_path)
        src_img = read_img(source_path, 0.5, 0.5, 'cuda')

        timer.tic()

        # Semantic Regularization
        if lam > 0:
            parse_map = net_parse((src_img + 1) / 2)[0]
            parse_map = parse_map.squeeze(0).argmax(0).unsqueeze(0).detach()

            parse_map[parse_map == 17] = 0  # denote hair as background, which we ignore in the loss

        # encode images
        cond = model.encode(src_img)

        xT_all = model.encode_stochastic_all(src_img, cond, T=T_enc)  # xT_all is [x_1, ..., x_T=Noise]

        xT = xT_all[start_t * (T_enc // T_atk) - 1]  # Make the stocastic noise consitent from the encoding step to the attack step

        if lam > 0:
            if vis_full:
                z_adv, adv_img, img_list = test_attacker.perturb(cond, target_embeding, xT, parse_map=parse_map)
            else:
                z_adv, adv_img = test_attacker.perturb(cond, target_embeding, xT, parse_map=parse_map)
        else:
            if vis_full:
                z_adv, adv_img, img_list = test_attacker.perturb(cond, target_embeding, xT)
            else:
                z_adv, adv_img = test_attacker.perturb(cond, target_embeding, xT)

        # reconstruct adv images
        xT = xT_all[-1]
        adv_render_img = model.render(xT, z_adv, T=T_inf)
        avg_time = timer.toc()
        time_list.append(avg_time)

        save_name = os.path.join(save_path, source_name)
        save_image(adv_render_img, save_name)

        # Full visualization
        if vis_full:
            vis_path = os.path.join(args.save_path, 'vis_full')
            os.makedirs(vis_path, exist_ok=True)
            source_name = source_name.split('.')[0]
            vis_input_name = os.path.join(vis_path, source_name + '_input.png')
            vis_encode_name = os.path.join(vis_path, source_name + '_encode.png')
            vis_adv_name = os.path.join(vis_path, source_name + '_adv.png')
            vis_input_parse_name = os.path.join(vis_path, source_name + '_input_parse.png')
            vis_adv_parse_name = os.path.join(vis_path, source_name + '_adv_parse.png')

            parse_map_input = net_parse((src_img + 1) / 2)[0]
            parse_map_input = parse_map_input.squeeze(0).cpu().detach().numpy().argmax(0)

            parse_map_adv = net_parse(adv_render_img)[0]
            parse_map_adv = parse_map_adv.squeeze(0).cpu().detach().numpy().argmax(0)

            save_image((src_img + 1) / 2, vis_input_name)
            save_image(xT, vis_encode_name)
            save_image(adv_render_img, vis_adv_name)
            vis_parsing_maps(Image.open(vis_input_name), parse_map_input, save_path=vis_input_parse_name)
            vis_parsing_maps(Image.open(vis_adv_name), parse_map_adv, save_path=vis_adv_parse_name)

            for i, x_tmp in enumerate(img_list):
                vis_tmp_name = os.path.join(vis_path, source_name + f'_{str(i)}.png')
                save_image((x_tmp + 1) / 2, vis_tmp_name)

            return

    print('Finished! Image saved in:', os.path.abspath(args.save_path))
    result_fn = os.path.join(args.save_path, "time.txt")
    f = open(result_fn, 'a')
    print('Time: ', round(np.average(time_list),2))
    f.write(f"Time: {round(np.average(time_list),2)}\n")

def attack_local_models(args, attack=True):
    test_models = load_test_models(args.test_model_names)
    th_dict = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
               'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878), 'cosface': (0.144840, 0.241045, 0.312703), 'arcface': (0.144840, 0.241045, 0.312703)}

    result_fn = os.path.join(args.save_path, "result.txt")
    f = open(result_fn, 'a')
    print('Is Adversarial Attack:', attack)
    f.write(f"Is Adversarial Attack: {attack}\n")

    combined_dir = os.path.join(args.save_path, "combined")
    os.makedirs(combined_dir, exist_ok=True)

    for test_model in test_models.keys():
        size = test_models[test_model][0]
        model = test_models[test_model][1]

        target = read_img(args.test_path, 0.5, 0.5, 'cuda')

        target_embbeding = model.forward((F.interpolate(target, size=size, mode='bilinear')))

        FAR01 = 0
        FAR001 = 0
        FAR0001 = 0
        total = 0
        if attack:
            for img_path in glob.glob(os.path.join(args.save_path, "*.png")):

                adv_example = read_img(img_path, 0.5, 0.5, 'cuda')
                ae_embbeding = model.forward((F.interpolate(adv_example, size=size, mode='bilinear')))
                fn = img_path.split("/")[-1]
                clean_img = cv2.imread(os.path.join(args.clean_path, fn))


                cos_simi = torch.cosine_similarity(ae_embbeding, target_embbeding)

                if cos_simi.item() > th_dict[test_model][0]:
                    FAR01 += 1
                if cos_simi.item() > th_dict[test_model][1]:
                    FAR001 += 1
                if cos_simi.item() > th_dict[test_model][2]:
                    FAR0001 += 1
                total += 1

                # combine the clean and adv image for visualization
                adv_img = cv2.imread(img_path)
                # fn = img_path.split("/")[-1]
                # clean_img = cv2.imread(os.path.join(args.clean_path, fn))
                if 'AMT' in args.save_path:
                    continue
                combined_img = np.concatenate([clean_img, adv_img], 1)
                combined_fn = f"{fn.split('.')[0]}_{cos_simi.item():.4f}.png"
                cv2.imwrite(os.path.join(combined_dir, combined_fn), combined_img)

        else:
            for img in tqdm(os.listdir(args.clean_path), desc=test_model + ' clean'):
                adv_example = read_img(os.path.join(args.clean_path, img), 0.5, 0.5, 'cuda')
                ae_embbeding = model.forward((F.interpolate(adv_example, size=size, mode='bilinear')))

                cos_simi = torch.cosine_similarity(ae_embbeding, target_embbeding)
                if cos_simi.item() > th_dict[test_model][0]:
                    FAR01 += 1
                if cos_simi.item() > th_dict[test_model][1]:
                    FAR001 += 1
                if cos_simi.item() > th_dict[test_model][2]:
                    FAR0001 += 1
                total += 1


        result_str = f"{test_model} ASR in FAR@0.1: {FAR01/total:.4f}, ASR in FAR@0.01: {FAR001/total:.4f}, ASR in FAR@0.001: {FAR0001/total:.4f}\n"
        print(result_str)
        f.write(result_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="assets/datasets/CelebA-HQ_align", help="path to source images")
    parser.add_argument("--save_path", default="assets/datasets/val", help="path to generated images")
    parser.add_argument("--clean_path", default="assets/datasets/CelebA-HQ_align", help="path to clean images")
    parser.add_argument("--target_path", default="assets/datasets/target/085807.jpg", help="path to target images")
    parser.add_argument("--test_path", default="assets/datasets/test/047073.jpg", help="path to test images")
    parser.add_argument('--device', type=str, default='0', help='cuda device')
    parser.add_argument("--model_path", default="checkpoints/G.pth", help="model for loading")
    parser.add_argument("--test_model_names", nargs='+', default=['mobile_face'], help="model for testing")
    parser.add_argument("--model_names", nargs='+', default=['ir152', 'irse50', 'facenet'], help="model for attacking")
    parser.add_argument("--eps", type=float, default=0.02, help="latent attack budget")
    parser.add_argument("--iter", type=int, default=50, help="Attack iterations (Global loop)")
    parser.add_argument("--T_enc", type=int, default=100, help="DDIM steps during image encoding")
    parser.add_argument("--T_atk", type=int, default=5, help="DDIM steps during attack")
    parser.add_argument("--T_inf", type=int, default=100, help="DDIM steps during inference")
    parser.add_argument("--start_t", type=int, default=5, help="starting point of attack")
    parser.add_argument("--attack_iter", type=int, default=1, help="number of attack iteration")
    parser.add_argument("--attack_inf_iter", type=int, default=4, help="number of attack inference iteration")
    parser.add_argument("--repeat_times", type=int, default=1, help="number of repeatance of attack inference and attack steps ")
    parser.add_argument("--cnt_skip", type=int, default=0, help="number of skip of attack inference and attack steps ")
    parser.add_argument("--lam", type=float, default=0., help="hyperparameter of face parsing ")
    parser.add_argument("--vis_full", action='store_true', default=False,
                        help="compare with other method")

    # T_atk = (attack_iter + attack_inf_iter) * repeat_times

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    args.save_path = os.path.join(args.save_path,
                                  f"save_eps{args.eps}_Tenc{args.T_enc}_iter{args.iter}_Tatk{args.T_atk}_Tstart{args.start_t}_Tinf{args.T_inf}"
                                  f"_atk{args.attack_iter}_atkinf{args.attack_inf_iter}_repeat{args.repeat_times}_skip{args.cnt_skip}_lam{args.lam}")

    generate(args)
    attack_local_models(args, attack=False)
    attack_local_models(args, attack=True)
