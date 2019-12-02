import torch
import torch.nn as nn
import numpy as np


def gram_matrix(mat):
  mat = mat.squeeze(dim=0)
  mat = torch.mm(mat, mat.t())
  return mat



def pairwise_dist(x, y):
  x_norm = (x.norm(dim=2)[:, :, None])
  y_t = y.permute(0, 2, 1).contiguous()
  y_norm = (y.norm(dim=2)[:, None])
  y_t = torch.cat([y_t] * x.shape[0], dim=0)
  dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
  return torch.clamp(dist, 0.0, np.inf)

# Model definition
class ShapeletGenerator(nn.Module):

  def __init__(self, n_prototypes, bag_size, n_classes):
    n_prototypes = int(n_prototypes)
    super(ShapeletGenerator, self).__init__()
    self.prototypes = (torch.randn(
        (1, n_prototypes, bag_size))).requires_grad_()
    if n_classes == 2:
      n_classes = 1
    self.linear_layer = torch.nn.Linear(3 * n_prototypes, n_classes, bias=False)
    #self.linear_layer.weight = torch.nn.Parameter(self.linear_layer.weight/100000)
    self.n_classes = n_classes

  def pairwise_distances(self, x, y):
    x_norm = (x.norm(dim=2)[:, :, None])
    y_t = y.permute(0, 2, 1).contiguous()
    y_norm = (y.norm(dim=2)[:, None])
    y_t = torch.cat([y_t] * x.shape[0], dim=0)
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)

  def get_output(self, batch_inp):
    dist = self.pairwise_distances(batch_inp, self.prototypes)
    min_dist = dist.min(dim=1)[0]
    max_dist = dist.max(dim=1)[0]
    mean_dist = dist.mean(dim=1)
    all_features = torch.cat([min_dist, max_dist, mean_dist], dim=1)
    logits = self.linear_layer(all_features)

    return logits, all_features

  def forward(self, x):
    logits, distances = self.get_output(x)
    if self.n_classes == 1:
      logits = logits.view(1)
    return logits, distances
