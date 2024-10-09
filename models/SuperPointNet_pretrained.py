"""
Network to load pretrained model from Magicleap.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
import torchvision


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )

class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self, BATCH_SIZE):
    super(SuperPointNet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Generate coordinates for rotation-invariant convolution
    self.coords11 = self.generate_coordinates(BATCH_SIZE, 128, 128)
    self.coords21 = self.generate_coordinates(BATCH_SIZE, 64, 64)
    self.coords31 = self.generate_coordinates(BATCH_SIZE, 32, 32)
    self.coords41 = self.generate_coordinates(BATCH_SIZE, 16, 16)

    # Shared Encoder.（特征提取部分）
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head. 检测头，检测
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

  def generate_coordinates(self, batch_size, height, width):
      coords = torch.zeros(height, width, 2 * 3 * 3)
      center = torch.tensor([height // 2, width // 2], dtype=torch.float32)
      grid_x, grid_y = torch.meshgrid(torch.arange(height), torch.arange(width))
      delta_x = grid_x - center[0]
      delta_y = grid_y - center[1]
      theta = torch.atan2(delta_y, delta_x) % (2 * torch.pi)

      for i in range(8):
          angle = theta + i * torch.pi / 4
          coords[:, :, 2 * i] = torch.cos(angle)
          coords[:, :, 2 * i + 1] = torch.sin(angle)

      coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1).permute(0, 3, 1, 2).cuda()
      return torch.autograd.Variable(coords, requires_grad=False)

  def deform_conv(self, x, conv_layer, coords):
      return torchvision.ops.deform_conv2d(input=x, offset=coords, weight=conv_layer.weight, padding=(1, 1))

  # def forward(self, x):
  #   """ Forward pass that jointly computes unprocessed point and descriptor
  #   tensors.
  #   Input
  #     x: Image pytorch tensor shaped N x 1 x H x W.
  #   Output
  #     semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
  #     desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
  #   """
  #   # Shared Encoder.
  #   x = self.relu(self.conv1a(x))
  #   x = self.relu(self.conv1b(x))
  #   x = self.pool(x)
  #   x = self.relu(self.conv2a(x))
  #   x = self.relu(self.conv2b(x))
  #   x = self.pool(x)
  #   x = self.relu(self.conv3a(x))
  #   x = self.relu(self.conv3b(x))
  #   x = self.pool(x)
  #   x = self.relu(self.conv4a(x))
  #   x = self.relu(self.conv4b(x))
  #   # Detector Head.
  #   cPa = self.relu(self.convPa(x))
  #   semi = self.convPb(cPa)
  #
  #   # Descriptor Head.
  #   cDa = self.relu(self.convDa(x))
  #   desc = self.convDb(cDa)
  #   dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
  #   desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.

  def forward(self, x):
      """ Forward pass with rotation-invariant convolutions. """
      x = self.relu(self.deform_conv(x, self.conv1a, self.coords11))
      x = self.relu(self.deform_conv(x, self.conv1b, self.coords11))
      x = self.pool(x)

      x = self.relu(self.deform_conv(x, self.conv2a, self.coords21))
      x = self.relu(self.deform_conv(x, self.conv2b, self.coords21))
      x = self.pool(x)

      x = self.relu(self.deform_conv(x, self.conv3a, self.coords31))
      x = self.relu(self.deform_conv(x, self.conv3b, self.coords31))
      x = self.pool(x)

      x = self.relu(self.deform_conv(x, self.conv4a, self.coords41))
      x = self.relu(self.deform_conv(x, self.conv4b, self.coords41))

      # Detector Head
      cPa = self.relu(self.deform_conv(x, self.convPa, self.coords41))
      semi = self.convPb(cPa)

      # Descriptor Head
      cDa = self.relu(self.deform_conv(x, self.convDa, self.coords41))
      desc = self.convDb(cDa)
      desc = desc / torch.norm(desc, p=2, dim=1, keepdim=True)

      return semi, desc









###############################
class PoseExpNet(nn.Module):

    def __init__(self, nb_ref_imgs=2, output_exp=False):
        super(PoseExpNet, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(3*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])

        self.pose_pred = nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0)

        if self.output_exp:
            upconv_planes = [256, 128, 64, 32, 16]
            self.upconv5 = upconv(conv_planes[4],   upconv_planes[0])
            self.upconv4 = upconv(upconv_planes[0], upconv_planes[1])
            self.upconv3 = upconv(upconv_planes[1], upconv_planes[2])
            self.upconv2 = upconv(upconv_planes[2], upconv_planes[3])
            self.upconv1 = upconv(upconv_planes[3], upconv_planes[4])

            self.predict_mask4 = nn.Conv2d(upconv_planes[1], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask3 = nn.Conv2d(upconv_planes[2], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask2 = nn.Conv2d(upconv_planes[3], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask1 = nn.Conv2d(upconv_planes[4], self.nb_ref_imgs, kernel_size=3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, target_image, ref_imgs):
        assert(len(ref_imgs) == self.nb_ref_imgs)
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1)
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)

        if self.output_exp:
            out_upconv5 = self.upconv5(out_conv5  )[:, :, 0:out_conv4.size(2), 0:out_conv4.size(3)]
            out_upconv4 = self.upconv4(out_upconv5)[:, :, 0:out_conv3.size(2), 0:out_conv3.size(3)]
            out_upconv3 = self.upconv3(out_upconv4)[:, :, 0:out_conv2.size(2), 0:out_conv2.size(3)]
            out_upconv2 = self.upconv2(out_upconv3)[:, :, 0:out_conv1.size(2), 0:out_conv1.size(3)]
            out_upconv1 = self.upconv1(out_upconv2)[:, :, 0:input.size(2), 0:input.size(3)]

            exp_mask4 = nn.functional.sigmoid(self.predict_mask4(out_upconv4))
            exp_mask3 = nn.functional.sigmoid(self.predict_mask3(out_upconv3))
            exp_mask2 = nn.functional.sigmoid(self.predict_mask2(out_upconv2))
            exp_mask1 = nn.functional.sigmoid(self.predict_mask1(out_upconv1))
        else:
            exp_mask4 = None
            exp_mask3 = None
            exp_mask2 = None
            exp_mask1 = None

        if self.training:
            return [exp_mask1, exp_mask2, exp_mask3, exp_mask4], pose
        else:
            return exp_mask1, pose
