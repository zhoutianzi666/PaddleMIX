import paddle
"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""
from collections import namedtuple
from taming.util import get_ckpt_path
from paddle.vision import models as tv


class LPIPS(paddle.nn.Layer):

    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.stop_gradient = not False

    def load_from_pretrained(self, name='vgg_lpips'):
        ckpt = get_ckpt_path(name, 'taming/modules/autoencoder/lpips')
        self.set_state_dict(state_dict=paddle.load(path=str(ckpt)))
        print('loaded pretrained LPIPS loss from {}'.format(ckpt))

    @classmethod
    def from_pretrained(cls, name='vgg_lpips'):
        if name != 'vgg_lpips':
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.set_state_dict(state_dict=paddle.load(path=str(ckpt)))
        return model

    def forward(self, input, target):
        in0_input, in1_input = self.scaling_layer(input), self.scaling_layer(
            target)
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]
                ), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for
            kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(paddle.nn.Layer):

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(name='shift', tensor=paddle.to_tensor(data=[-
            0.03, -0.088, -0.188], dtype='float32')[None, :, None, None])
        self.register_buffer(name='scale', tensor=paddle.to_tensor(data=[
            0.458, 0.448, 0.45], dtype='float32')[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(paddle.nn.Layer):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [paddle.nn.Dropout()] if use_dropout else []
        layers += [paddle.nn.Conv2D(in_channels=chn_in, out_channels=
            chn_out, kernel_size=1, stride=1, padding=0, bias_attr=False)]
        self.model = paddle.nn.Sequential(*layers)


class vgg16(paddle.nn.Layer):

    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = paddle.nn.Sequential()
        self.slice2 = paddle.nn.Sequential()
        self.slice3 = paddle.nn.Sequential()
        self.slice4 = paddle.nn.Sequential()
        self.slice5 = paddle.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_sublayer(name=str(x), sublayer=
                vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_sublayer(name=str(x), sublayer=
                vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_sublayer(name=str(x), sublayer=
                vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_sublayer(name=str(x), sublayer=
                vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_sublayer(name=str(x), sublayer=
                vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.stop_gradient = not False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2',
            'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3
            )
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = paddle.sqrt(x=paddle.sum(x=x ** 2, axis=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean(axis=[2, 3], keepdim=keepdim)
