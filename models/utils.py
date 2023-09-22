import torch
import geffnet
import torch.nn as nn


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class deepFeatureExtractor_EfficientNet(nn.Module):
    def __init__(self, architecture="EfficientNet-B5", lv6=False, lv5=False, lv4=False, lv3=False):
        super(deepFeatureExtractor_EfficientNet, self).__init__()
        assert architecture in ["EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2", "EfficientNet-B3", 
                                    "EfficientNet-B4", "EfficientNet-B5", "EfficientNet-B6", "EfficientNet-B7"]
        
        if architecture == "EfficientNet-B0":
            self.encoder = geffnet.tf_efficientnet_b0_ns(pretrained=True)
            self.dimList = [16, 24, 40, 112, 1280] #5th feature is extracted after conv_head or bn2
            #self.dimList = [16, 24, 40, 112, 320] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B1":
            self.encoder = geffnet.tf_efficientnet_b1_ns(pretrained=True)
            self.dimList = [16, 24, 40, 112, 1280] #5th feature is extracted after conv_head or bn2
            #self.dimList = [16, 24, 40, 112, 320] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B2":
            self.encoder = geffnet.tf_efficientnet_b2_ns(pretrained=True)
            self.dimList = [16, 24, 48, 120, 1408] #5th feature is extracted after conv_head or bn2
            #self.dimList = [16, 24, 48, 120, 352] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B3":
            self.encoder = geffnet.tf_efficientnet_b3_ns(pretrained=True)
            self.dimList = [24, 32, 48, 136, 1536] #5th feature is extracted after conv_head or bn2
            #self.dimList = [24, 32, 48, 136, 384] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B4":
            self.encoder = geffnet.tf_efficientnet_b4_ns(pretrained=True)
            self.dimList = [24, 32, 56, 160, 1792] #5th feature is extracted after conv_head or bn2
            #self.dimList = [24, 32, 56, 160, 448] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B5":
            self.encoder = geffnet.tf_efficientnet_b5_ns(pretrained=True)
            self.dimList = [24, 40, 64, 176, 2048] #5th feature is extracted after conv_head or bn2
            #self.dimList = [24, 40, 64, 176, 512] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B6":
            self.encoder = geffnet.tf_efficientnet_b6_ns(pretrained=True)
            self.dimList = [32, 40, 72, 200, 2304] #5th feature is extracted after conv_head or bn2
            #self.dimList = [32, 40, 72, 200, 576] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B7":
            self.encoder = geffnet.tf_efficientnet_b7_ns(pretrained=True)
            self.dimList = [32, 48, 80, 224, 2560] #5th feature is extracted after conv_head or bn2
            #self.dimList = [32, 48, 80, 224, 640] #5th feature is extracted after blocks[6]
        del self.encoder.global_pool
        del self.encoder.classifier
        #self.block_idx = [3, 4, 5, 7, 9] #5th feature is extracted after blocks[6]
        #self.block_idx = [3, 4, 5, 7, 10] #5th feature is extracted after conv_head
        self.block_idx = [3, 4, 5, 7, 11] #5th feature is extracted after bn2
        if lv6 is False:
            del self.encoder.blocks[6]
            del self.encoder.conv_head
            del self.encoder.bn2
            del self.encoder.act2
            self.block_idx = self.block_idx[:4]
            self.dimList = self.dimList[:4]
        if lv5 is False:
            del self.encoder.blocks[5]
            self.block_idx = self.block_idx[:3]
            self.dimList = self.dimList[:3]
        if lv4 is False:
            del self.encoder.blocks[4]
            self.block_idx = self.block_idx[:2]
            self.dimList = self.dimList[:2]
        if lv3 is False:
            del self.encoder.blocks[3]
            self.block_idx = self.block_idx[:1]
            self.dimList = self.dimList[:1]
        # after passing blocks[3]    : H/2  x W/2
        # after passing blocks[4]    : H/4  x W/4
        # after passing blocks[5]    : H/8  x W/8
        # after passing blocks[7]    : H/16 x W/16
        # after passing conv_stem    : H/32 x W/32
        self.fixList = ['blocks.0.0','bn']

        for name, parameters in self.encoder.named_parameters():
            if name == 'conv_stem.weight':
                parameters.requires_grad = False
            if any(x in name for x in self.fixList):
                parameters.requires_grad = False
        
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        block_cnt = 0
        for k, v in self.encoder._modules.items():
            if k == 'act2':
                break
            if k == 'blocks':
                for m, n in v._modules.items():
                    feature = n(feature)
                    try:
                        if self.block_idx[block_cnt] == cnt:
                            out_featList.append(feature)
                            block_cnt += 1
                            break
                        cnt += 1
                    except:
                        continue
            else:
                feature = v(feature)
                if self.block_idx[block_cnt] == cnt:
                    out_featList.append(feature)
                    block_cnt += 1
                    break
                cnt += 1            
            
        return out_featList

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable
