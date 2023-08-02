from templates import *
import torch
from assets.models import irse, ir152, facenet
from advertorch.utils import NormalizeByChannelMeanStd
from iterative_projected_gradient import PGDAttack
from advertorch.context import ctx_noparamgrad_and_eval
from torchvision.transforms import ToPILImage, ToTensor

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

device = 'cuda:0'
conf = ffhq256_autoenc()
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

data = ImageDataset('imgs_align', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
batch = data[1]['img'][None]

cond = model.encode(batch.to(device))
xT = model.encode_stochastic(batch.to(device), cond, T=250)


model_names = ['ir152', 'irse50', 'facenet']
th_dict = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
               'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878)}

test_models = {}
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


net = Net(test_models, model).to(device)
test_attacker = PGDAttack(predict=net,
                          loss_fn=Cos_Loss,
                          eps=0.1,
                          eps_iter=0.02,
                          nb_iter=10,
                          clip_min=-1e6,
                          clip_max=1e6,
                          targeted=True,
                          rand_init=False)


target = data[0]['img'][None]

target_embeding = net(target.to(device))

z_adv = test_attacker.perturb(cond, target_embeding, xT, T=5)

