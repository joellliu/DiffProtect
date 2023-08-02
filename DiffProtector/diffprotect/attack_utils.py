from assets.models import irse, ir152, facenet
import torch
import cv2
from advertorch.utils import NormalizeByChannelMeanStd
import torch.nn.functional as F
import torch.nn as nn

def preprocess(im, mean, std, device):
    if len(im.size()) == 3:
        im = im.transpose(0, 2).transpose(1, 2).unsqueeze(0)
    elif len(im.size()) == 4:
        im = im.transpose(1, 3).transpose(2, 3)

    mean = torch.tensor(mean).to(device)
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std).to(device)
    std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    im = (im - mean) / std
    return im


def read_img(data_dir, mean, std, device):
    img = cv2.imread(data_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    img = torch.from_numpy(img).to(torch.float32).to(device)
    img = preprocess(img, mean, std, device)
    return img

class Net(torch.nn.Module):
    def __init__(self, test_models, decoder=None):
        super(Net, self).__init__()
        self.test_models = test_models
        self.decoder = decoder
        self.norm = NormalizeByChannelMeanStd([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).cuda()

    def forward(self, z, xT=None, T=1):

        if xT is None:  # input is images
            x = z
        else:  # input are latent codes
            #x = self.decoder.render(xT, z, T)
            x = self.decoder.render(z, xT, T)

        x = self.norm(x)
        features = []
        for model_name in self.test_models.keys():
            input_size = self.test_models[model_name][0]
            fr_model = self.test_models[model_name][1]
            source_resize = F.interpolate(x, size=input_size, mode='bilinear')
            emb_source = fr_model(source_resize)
            features.append(emb_source)
        # avg_feature = torch.mean(torch.stack(features), dim=0)
        avg_feature = features

        return avg_feature


def cos_simi(emb_1, emb_2):
    return torch.mean(torch.sum(torch.mul(emb_2, emb_1), dim=1) / emb_2.norm(dim=1) / emb_1.norm(dim=1))

def Cos_Loss(source_feature, target_feature):
    cos_loss_list = []
    for i in range(len(source_feature)):
        cos_loss_list.append(1 - cos_simi(source_feature[i], target_feature[i].detach()))
        # print(1 - cos_simi(source_feature[i], target_feature[i]))
    cos_loss = torch.mean(torch.stack(cos_loss_list))
    return cos_loss


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


def load_test_models(model_names):
    test_models = {}
    device='cuda'
    for model_name in model_names:
        if model_name == 'ir152':
            test_models[model_name] = []
            test_models[model_name].append((112, 112))
            fr_model = ir152.IR_152((112, 112))
            fr_model.load_state_dict(torch.load('./assets/models/ir152.pth'))
            fr_model.to(device)
            fr_model.eval()
            test_models[model_name].append(fr_model)
        if model_name == 'irse50':
            test_models[model_name] = []
            test_models[model_name].append((112, 112))
            fr_model = irse.Backbone(50, 0.6, 'ir_se')
            fr_model.load_state_dict(torch.load('./assets/models/irse50.pth'))
            fr_model.to(device)
            fr_model.eval()
            test_models[model_name].append(fr_model)
        if model_name == 'facenet':
            test_models[model_name] = []
            test_models[model_name].append((160, 160))
            fr_model = facenet.InceptionResnetV1(num_classes=8631, device=device)
            fr_model.load_state_dict(torch.load('./assets/models/facenet.pth'))
            fr_model.to(device)
            fr_model.eval()
            test_models[model_name].append(fr_model)
        if model_name == 'mobile_face':
            test_models[model_name] = []
            test_models[model_name].append((112, 112))
            fr_model = irse.MobileFaceNet(512)
            fr_model.load_state_dict(torch.load('./assets/models/mobile_face.pth'))
            fr_model.to(device)
            fr_model.eval()
            test_models[model_name].append(fr_model)
    return test_models


class Net(torch.nn.Module):
    def __init__(self, test_models, decoder=None):
        super(Net, self).__init__()
        self.test_models = test_models
        self.decoder = decoder
        self.norm = NormalizeByChannelMeanStd([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).cuda()

    def forward(self, z, xT=None, T=1):

        if xT is None:  # input is images
            x = z
        else:  # input are latent codes
            # x = self.decoder.render(xT, z, T)
            x = self.decoder.render(z, xT, T)
            # print(x.size())

        x = self.norm(x)
        features = []
        for model_name in self.test_models.keys():
            input_size = self.test_models[model_name][0]
            fr_model = self.test_models[model_name][1]
            source_resize = F.interpolate(x, size=input_size, mode='bilinear')
            emb_source = fr_model(source_resize)
            features.append(emb_source)
        # avg_feature = torch.mean(torch.stack(features), dim=0)
        avg_feature = features
        # print(len(avg_feature), avg_feature[0].size())

        return avg_feature


class Net_fast(torch.nn.Module):
    def __init__(self, test_models, decoder=None):
        super(Net_fast, self).__init__()
        self.test_models = test_models
        self.decoder = decoder
        self.norm = NormalizeByChannelMeanStd([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).cuda()
        # self.model = self.decoder.ema_model

    def forward(self, z, xT=None, T=1, t=None, predict=True, is_last=False):

        if xT is None:  # input is images
            x = z
            x_pred = z
        else:  # input are latent codes
            # x = self.decoder.render(xT, z, T)
            # x = self.decoder.render(z, xT, T)
            x, x_pred = self.decoder.render(z, xT, T, True, t, is_last)
            # print(x.size())
            # x = out["pred_xstart"]
            # x_pred = out["sample"]

        if predict:
            # if is_last:
            #     x_norm = self.norm(x_pred)
            # else:
            #     x_norm = self.norm(x)
            x_norm = self.norm(x)
            # x_norm = x_pred
            features = []
            for model_name in self.test_models.keys():
                input_size = self.test_models[model_name][0]
                fr_model = self.test_models[model_name][1]
                source_resize = F.interpolate(x_norm, size=input_size, mode='bilinear')
                emb_source = fr_model(source_resize)
                features.append(emb_source)
            # avg_feature = torch.mean(torch.stack(features), dim=0)
            avg_feature = features
            # print(len(avg_feature), avg_feature[0].size())

            return avg_feature, x_pred, x

        else:
            return x_pred, x

    def encode_all(self, z, x, T=1):

        xT = self.decoder.encode_stochastic_all(x, z, T)

        return xT

    # def encode_t(self, z, x, T=1):
    #
    #     xT = self.decoder.encode_stochastic_all(x, z, T)
    #
    #     return xT
