#     Copyright 2020 Google LLC
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#         https://www.apache.org/licenses/LICENSE-2.0
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import torch.nn as nn
from torch.nn.init import xavier_uniform_, kaiming_uniform_, zeros_, kaiming_normal_
import torch
from .wideresnet import WideResNet


def add_model_opts(parser):
	parser.add_argument('-init-method', type=str, default='xavier_unif', choices=['xavier_unif', 'kaiming_unif'])
	parser.add_argument('-loss-fn', type=str, default='CE', choices=['CE', 'BCE', 'MSE'])
	# For WideResnet Model
	parser.add_argument('-depth', type=int, default=22)
	parser.add_argument('-widen-factor', type=int, default=4)
	parser.add_argument('-dropRate', type=float, default=0.1)
	parser.add_argument('-ft-dropRate', type=float, default=0.1)
	return parser


# Recursively delete attribute and its children
def del_attr(obj, names):
	if len(names) == 1:
		delattr(obj, names[0])
	else:
		del_attr(getattr(obj, names[0]), names[1:])


# Recursively set attribute and its children
def set_attr(obj, names, val):
	if len(names) == 1:
		setattr(obj, names[0], val)
	else:
		set_attr(getattr(obj, names[0]), names[1:], val)


# Weight initialization
def weight_init(init_method):

	def initfn(layer):
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			if init_method == 'xavier_unif':
				xavier_uniform_(layer.weight.data)
			elif init_method == 'kaiming_unif':
				kaiming_uniform_(layer.weight.data)
			elif init_method == 'kaiming_normal':
				kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
			if layer.bias is not None:
				zeros_(layer.bias.data)
		elif isinstance(layer, nn.BatchNorm2d):
			layer.weight.data.fill_(1)
			layer.bias.data.zero_()
	return initfn


# Super-class encapsulating all model related functions
class Model(nn.Module):
	def __init__(
					self, loss_name='CE'
				):
		super(Model, self).__init__()
		self.loss_fn_name = loss_name
		self.loss_fn = self.get_loss_fn(loss_name)

	def get_loss_fn(self, fn_name, reduction='mean'):
		if fn_name == 'CE':
			return nn.CrossEntropyLoss(reduction=reduction)
		elif fn_name == 'BCE':
			return nn.BCELoss(reduction=reduction)
		elif fn_name == 'MSE':
			return nn.MSELoss(reduction=reduction)

	def forward(self, x, head_name=None, body_only=False, head_only=False):
		assert not (head_only and body_only), 'Cannot have both head only and body only enabled'
		if body_only:
			return self.model.apply_body(x)
		elif head_only:
			return self.model.apply_head(x, head_name=head_name)
		m_out = self.model(x, head_name=head_name)
		if self.loss_fn_name == 'BCE':
			# We need to do a sigmoid if we're using binary labels
			m_out = torch.sigmoid(m_out)
		return m_out

	def criterion(self, outs, target):
		if self.loss_fn_name == 'BCE':
			target = target.float().unsqueeze(1)
		return self.loss_fn(outs, target)


class WideResnet(Model):
	def __init__(
					self, out_class_dict, depth, widen_factor,
					loss_name='CE', dropRate=0.0
				):
		super(WideResnet, self).__init__(
									loss_name=loss_name,
								)
		self.model = WideResNet(depth, out_class_dict, widen_factor=widen_factor, dropRate=dropRate)
		self.model.apply(weight_init('kaiming_normal'))

	def add_heads(self, class_dict, is_cuda=True):
		self.model.add_heads(class_dict, is_cuda=is_cuda, init_fn=weight_init('kaiming_normal'))
