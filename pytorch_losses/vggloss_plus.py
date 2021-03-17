import torch
import torch.nn as nn
from torchvision import models

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out

def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

class VGGLoss_plus(nn.Module):
    def __init__(self, device):
        super(VGGLoss_plus, self).__init__()

        self.vgg = Vgg16().to(device)

        self.criterion = nn.MSELoss() #nn.L1Loss()
#         self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]     
        self.weights = [1.0, 1.0, 1.0, 1.0]     

    def forward(self, x, y, compare_all=True):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        
        #ms-content_loss
        content_loss = 0
        if compare_all:
            for i in range(len(x_vgg)):
                content_loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        else:
            for i in range(len(x_vgg) - 1, len(x_vgg) - 3, -1):
                content_loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
                
        #ms-style_loss
        x_gram = [gram(fmap) for fmap in x_vgg]
        y_gram = [gram(fmap) for fmap in y_vgg]
        style_loss = 0
        if compare_all:
            for i in range(len(x_gram)):
                style_loss += self.weights[i] * self.criterion(x_gram[i], y_gram[i].detach()) 
        else:
            for i in range(len(x_gram) - 1, len(x_gram) - 3, -1):
                style_loss += self.weights[i] * self.criterion(x_gram[i], y_gram[i].detach()) 
               
        return content_loss, style_loss