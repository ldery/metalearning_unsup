import numpy as np
from sklearn.linear_model import Ridge
import pdb
from copy import deepcopy
import random
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from sklearn.metrics import r2_score
import torch.functional as F


def set_seed(seed):
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)


def get_problem(d, n_tr, n_ts, n_aux, s_x, s_e1, s_e2):
	def get_xy_pair(n):
		x_tr = np.random.normal(scale=s_x, size=(n, d))
		tr_noise = np.random.normal(scale=s_e1, size=(n, 1))
		y_tr = np.matmul(x_tr, w_opt.T) + np.random.normal(scale=s_e2, size=(n, 1))
		x_tr = np.hstack((x_tr, tr_noise))
		return x_tr, y_tr

	w_opt = np.random.normal(scale=s_x, size=(1, d))
	x_aux, y_aux = get_xy_pair(n_aux)
	# overwrite the last dimension
	x_aux[:, -1:] = y_aux[:]

	return w_opt, get_xy_pair(n_tr), get_xy_pair(n_ts), x_aux


class UnsupNet(nn.Module):
	def __init__(self):
		super(UnsupNet, self).__init__()
		network = {
			'i': 101,
			'h': 512,
			'o': 1,
			'l': 3
		}
		# rep_extractor
		self.rep_extractor = nn.Sequential(
				nn.Linear(network['i'], network['h']),
				nn.ReLU(),
				nn.Linear(network['h'], network['h']),
				nn.ReLU(),
			)
		# primary_head
		self.prim_head = nn.Linear(network['h'], network['o'])

		# aux_head
		self.aux_model1 = nn.Sequential(
				nn.Linear(network['h'], network['h']),
				nn.ReLU(),
				nn.Linear(network['h'], network['h']),
			)
		self.aux_model2 = nn.Linear(network['h'], network['h'])

	def get_optims(self, lr, wd):
		rep_optim = AdamW(self.rep_extractor.parameters(), lr=lr, weight_decay=wd)
		aux_params = tuple(*self.aux_model.parameters(), *self.aux_model2.parameters())
		aux_optim = AdamW(aux_params, lr=lr, weight_decay=wd)
		prim_optim = AdamW(self.prim_head.parameters(), lr=lr, weight_decay=wd)

		return rep_optim, aux_optim, prim_optim

	# should we just directly encode the loss function ??
	def forward(self, x, type_=None):
		rep = self.rep_extractor(x)
		if type_ == 'prim':
			return self.prim_head(rep)
		elif type_ == 'aux':
			out = self.aux_model1(rep)
			out = F.relu(out + rep)
			out = self.aux_model2(out)
			return out

def train_nn_meta(tr_x, tr_y, ts_x, ts_y, aux, lr=1e-3, n_epochs=10, wd=0.1, phase_=10):

	# perform data splitting
	n_tr = int(len(tr_x) * 0.8)
	aux = torch.tensor(aux).float()
	ts_x, ts_y = torch.tensor(ts_x).float(), torch.tensor(ts_y).float()
	tr_x, tr_y = torch.tensor(tr_x).float(), torch.tensor(tr_y).float()
	val_x, val_y = tr_x[-n_tr:, :], tr_y[-n_tr:, :]
	tr_x, tr_y = tr_x[:n_tr, :], tr_y[:n_tr, :]

	loss_fn = nn.MSELoss()
	model = UnsupNet()
	rep_optim, aux_optim, prim_optim = model.get_optims(lr, wd)
	all_optim = AdamW(model.parameters(), lr=0, weight_decay=0)

	def make_n_run(this_x, this_y, aux_x):
		# keep track of the required statistics
		dev_r2s, test_r2s = [], []
		model_cpy = None
		for i in range(n_epochs):
			# For n iterates, keep the body and aux-model fixed and learn a primary task head
			model.train()
			for j in range(phase_):
				pred = model(this_x, type_='prim')
				loss = loss_fn(pred, this_y)
				loss.backward()
				prim_optim.step()
				# Reset all the gradients
				all_optim.zero_grad()

			# For n iterates, keep the body and primary head fixed and learn a auxiliary model head
			# Todo - lucio - might need to clip gradients here and there
			for j in range(phase_):
				# This is the hard part
				# Right now I am approximating the gradient @ the next iterate with the gradient @ the current point
				pred = model(this_x, type_='prim')
				loss = loss_fn(pred, this_y)
				rep_prim_grad = torch.autograd.grad(loss, model.rep_extractor.parameters())

				# TODO : LDERY
				with torch.no_grad():
					q = model(aux_x, type_='aux')

				aux_optim.step()
				# Reset all the gradients
				all_optim.zero_grad()

			# For n iterates, keep train body and keep rest fixed
			for j in range(phase_):
				pred = model(this_x, type_='prim')
				prim_loss = loss_fn(pred, this_y)
				prim_loss.backward()

				# Need to check to make sure this is correct
				with torch.no_grad():
					daux_dauxhead = model(aux_x, type_='aux')
				body = model.rep_extractor(aux_x)
				aux_loss = (body * daux_dauxhead).sum(axis=-1).mean()
				aux_loss.backward()  # to get daux_dbody
				rep_optim.step()
				# Reset all the gradients
				all_optim.zero_grad()

			# do evaluation here
			model.eval()
			dev_r2 = r2_score(val_y.numpy(), model(val_x, type_='prim').detach().numpy())
			test_r2 = r2_score(ts_y.numpy(), model(ts_x, type_='prim').detach().numpy())
			dev_r2s.append(dev_r2)
			print('Iter {} | Dev Score {} | Test Score {} '.format(i, dev_r2, test_r2))
			if np.argmax(dev_r2s) == i:
				# We have reached a good point
				model_cpy = deepcopy(model)
			test_r2s.append(test_r2)

		model_cpy.eval()
		aux_y = model_cpy(aux_x, type_='prim').detach()
		return dev_r2s, test_r2s, aux_y



def train_nn(tr_x, tr_y, ts_x, ts_y, aux, lr=1e-3, n_epochs=10, wd=0.1):

	network = {
		'i': 101,
		'h': 512,
		'o': 1,
		'l': 3
	}

	# perform data splitting
	n_tr = int(len(tr_x) * 0.8)
	aux = torch.tensor(aux).float()
	ts_x, ts_y = torch.tensor(ts_x).float(), torch.tensor(ts_y).float()
	tr_x, tr_y = torch.tensor(tr_x).float(), torch.tensor(tr_y).float()
	val_x, val_y = tr_x[-n_tr:, :], tr_y[-n_tr:, :]
	tr_x, tr_y = tr_x[:n_tr, :], tr_y[:n_tr, :]

	def make_n_run(this_x, this_y, aux_x):
		# create the model
		layers = []
		i_, h_ = network['i'], network['h']
		for l_id in range(network['l']):
			if l_id == (network['l'] - 1):
				h_ = network['o']
			layers.append(nn.Linear(i_, h_))
			layers.append(nn.ReLU())
			i_ = h_

		model = nn.Sequential(*layers[:-1])
		loss_fn = nn.MSELoss()

		# we can just do full batched gradient descent here
		optim = Adam(model.parameters(), lr=lr)  # , weight_decay=wd)
		dev_r2s, test_r2s = [], []
		model_cpy = None
		for i in range(n_epochs):
			model.train()
			pred = model(this_x)
			loss = loss_fn(pred, this_y)
			loss.backward()
			optim.step()
			optim.zero_grad()
			# do evaluation here
			model.eval()
			dev_r2 = r2_score(val_y.numpy(), model(val_x).detach().numpy())
			test_r2 = r2_score(ts_y.numpy(), model(ts_x).detach().numpy())
			dev_r2s.append(dev_r2)
			print('Iter {} | Dev Score {} | Test Score {} '.format(i, dev_r2, test_r2))
			if np.argmax(dev_r2s) == i:
				# We have reached a good point
				model_cpy = deepcopy(model)
			test_r2s.append(test_r2)

		model_cpy.eval()
		aux_y = model_cpy(aux_x).detach()
		return dev_r2s, test_r2s, aux_y

	dev_r2s, test_r2s, aux_y = make_n_run(tr_x, tr_y, aux)
	best_idx = np.argmax(dev_r2s)
	best_r2 = test_r2s[best_idx]
	print('		This is the R^2 = {:.5f} at epoch {}'.format(best_r2, best_idx))
	print('		Performing Self Training')

	# tr_x, tr_y = torch.vstack((tr_x, aux)), torch.vstack((tr_y, aux_y))
	# dev_r2s, test_r2s, aux_y = make_n_run(tr_x, tr_y, aux)
	# best_idx = np.argmax(dev_r2s)
	# best_r2 = test_r2s[best_idx]
	# print('		This is the R^2 = {:.5f} at epoch {}'.format(best_r2, best_idx))



problem = {
	'd': 100,
	'n_tr': 60,
	'n_ts': 500,
	'n_aux': 200,
	's_x': 2,
	's_e1': 2,
	's_e2': 5,
}

set_seed(0)
w_opt, xy_tr, xy_ts, x_aux = get_problem(*problem.values())

print('Training Linear Model')
for alpha in [10, 20]:
	set_seed(0)
	clf = Ridge(alpha=alpha, fit_intercept=False).fit(xy_tr[0], xy_tr[1])
	score = r2_score(xy_ts[1], clf.predict(xy_ts[0]))
	print('This is when alpha = ', alpha)
	print('This is the R^2 = {:.5f} for naive training'.format(score))
	aux_pred = clf.predict(x_aux)
	new_x, new_y = np.vstack((xy_tr[0], x_aux)), np.vstack((xy_tr[1], aux_pred))

	# clf = Ridge(alpha=alpha, fit_intercept=False).fit(new_x, new_y)
	# score = clf.score(xy_ts[0], xy_ts[1])
	# print('This is the R^2 = {:.5f} for self-training'.format(score))


print('Training 3 Layer NN Model')
for lr in [1e-3, 7e-3, 1e-2, 3e-2]:
	for wd in [0, 1, 0.1, 0.01]:
		print('This is when lr = ', lr, ' wd = ', wd)
		# try:
		set_seed(0)
		train_nn(xy_tr[0], xy_tr[1], xy_ts[0], xy_ts[1], x_aux, lr=lr, n_epochs=200, wd=wd)
		exit()
		# except:
		# 	print('		Issue arose')

