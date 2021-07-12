import numpy as np
from sklearn.linear_model import Ridge
from copy import deepcopy
import random
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from sklearn.metrics import r2_score
import higher
import torch
from torch.utils.tensorboard import SummaryWriter


'''
Notes : 
1. The non-monotonicity of the auxiliary loss is important. Forcing the auxiliary loss to be >=0 yields less good results
2. The resnet like architecture did better than the non-resnet architecture 
3. Self-training improves perforformance
4. Self-training with the unsupervised objective does better than without
5. Doesn't work if you take out label info from embedding
'''

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
	# x_aux[:, -1:] = y_aux[:]

	return w_opt, get_xy_pair(n_tr), get_xy_pair(n_ts), x_aux


class ResNet(nn.Module):
	def __init__(self, h, o):
		super(ResNet, self).__init__()
		self.fc1 = nn.Linear(h, h)
		self.fc2 = nn.Linear(h, h)
		self.fc3 = nn.Linear(h, o, bias=False)

	def forward(self, inp):
		out = F.relu(self.fc1(inp))
		out = F.relu(self.fc2(inp + out))
		return self.fc3(out)  # torch.tanh()


class UnsupNet(nn.Module):
	def __init__(self, network):
		super(UnsupNet, self).__init__()
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
		self.aux_model = ResNet(network['h'], network['o'])

	def get_optims(self, lr, wd):
		rep_optim = Adam(self.rep_extractor.parameters(), lr=lr)  # , weight_decay=wd)
		aux_optim = AdamW(self.aux_model.parameters(), lr=lr, weight_decay=wd)
		prim_optim = Adam(self.prim_head.parameters(), lr=lr)  # , weight_decay=wd)

		return rep_optim, aux_optim, prim_optim

	# should we just directly encode the loss function ??
	def forward(self, x, type_=None):
		rep = self.rep_extractor(x)
		if type_ == 'prim':
			return self.prim_head(rep)
		elif type_ == 'aux':
			out = self.aux_model(rep)
			return out.mean()


def norm_(list_of_tensors):
	norm = 0
	for x in list_of_tensors:
		if x is not None:
			norm += (x**2).sum().item()
	return norm


def make_n_run_meta(
						model, d_tr, d_val, aux_x, d_ts,
						n_epochs, phase_, loss_fn, optims, writer,
						id_='main'
					):
	this_x, this_y = d_tr
	ts_x, ts_y = d_ts
	val_x, val_y = d_val
	rep_optim, prim_optim, aux_optim, all_optim = optims
	# keep track of the required statistics
	dev_r2s, test_r2s = [], []
	model_cpy = None
	for i in range(n_epochs):
		# For n iterates, keep the body and aux-model fixed and learn a primary task head
		model.train()
		avg_prim_loss = 0.0
		for j in range(phase_):
			pred = model(this_x, type_='prim')
			loss = loss_fn(pred, this_y)
			loss.backward()
			avg_prim_loss += loss.item()
			prim_optim.step()
			# Reset all the gradients
			all_optim.zero_grad()

		writer.add_scalar("{}/primloss_train.head".format(id_), avg_prim_loss / phase_, i)

		# For n iterates, keep the body and primary head fixed and learn a auxiliary model head
		# Todo - lucio - might need to clip gradients here and there

		for j in range(phase_):
			# This is the hard part
			with higher.innerloop_ctx(model, rep_optim, track_higher_grads=True) as (fmodel, diff_repoptim):
				this_auxloss = fmodel(aux_x, type_='aux')
				diff_repoptim.step(this_auxloss)

				prim_loss = loss_fn(fmodel(this_x, type_='prim'), this_y)
				all_grads = torch.autograd.grad(prim_loss, fmodel.parameters(time=0), allow_unused=True)
				# need to filter out
			aux_grads = all_grads[6:]
			for p, g in zip(model.aux_model.parameters(), aux_grads):
				if p.grad is None:
					p.grad = torch.zeros_like(p)
				p.grad.mul(torch.tensor([0]).float())
				p.grad.add_(g)

			aux_optim.step()
			# Reset all the gradients
			all_optim.zero_grad()

		# For n iterates, keep train body and keep rest fixed
		avg_prim_loss = 0.0
		avg_aux_loss = 0.0
		for j in range(phase_):
			pred = model(this_x, type_='prim')
			prim_loss = loss_fn(pred, this_y)
			prim_loss.backward()
			# get statistics

			avg_prim_loss += prim_loss.item()

			aux_loss = model(aux_x, type_='aux')
			aux_loss.backward()  # to get daux_dbody
			avg_aux_loss += aux_loss.item()
			rep_optim.step()
			# Reset all the gradients
			all_optim.zero_grad()

		writer.add_scalar("{}/primloss_train.body".format(id_), avg_prim_loss / phase_, i)
		writer.add_scalar("{}/aux_train.body".format(id_), avg_aux_loss / phase_, i)

		# do evaluation here
		model.eval()
		train_r2 = r2_score(this_y.numpy(), model(this_x, type_='prim').detach().numpy())
		dev_r2 = r2_score(val_y.numpy(), model(val_x, type_='prim').detach().numpy())
		test_r2 = r2_score(ts_y.numpy(), model(ts_x, type_='prim').detach().numpy())
		dev_r2s.append(dev_r2)
		test_r2s.append(test_r2)

		writer.add_scalars(
							"{}/r2_score".format(id_),
							{'train': train_r2, 'dev': dev_r2, 'test': test_r2},
							i
						)
		# aux_score = model(aux_x, type_='aux').item()
		# print('Iter {} | Dev Score {:.3f} | Test Score {:.3f} | Aux Score {:.3f}'.format(i, dev_r2, test_r2, aux_score))
		if np.argmax(dev_r2s) == i:
			# We have reached a good point
			model_cpy = deepcopy(model)

	model_cpy.eval()
	aux_y = model_cpy(aux_x, type_='prim').detach()
	return dev_r2s, test_r2s, aux_y


def make_n_run(network, d_tr, d_val, aux_x, d_ts, wd, lr, epochs=200):
	this_x, this_y = d_tr
	ts_x, ts_y = d_ts
	val_x, val_y = d_val

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
	optim = AdamW(model.parameters(), lr=lr, weight_decay=wd)
	dev_r2s, test_r2s = [], []
	model_cpy = None
	for i in range(epochs):
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
		test_r2s.append(test_r2)
		print('Iter {} | Dev Score {} | Test Score {} '.format(i, dev_r2, test_r2))

		if np.argmax(dev_r2s) == i:
			# We have reached a good point
			model_cpy = deepcopy(model)

	model_cpy.eval()
	aux_y = model_cpy(aux_x).detach()
	return dev_r2s, test_r2s, aux_y


def train_nn(
					tr_x, tr_y, ts_x, ts_y, aux, lr=1e-3,
					n_epochs=10, wd=0.1, phase_=10, is_meta=False,
					writer=None
				):

	# perform data splitting
	n_tr = int(len(tr_x) * 0.8)
	aux = torch.tensor(aux).float()
	ts_x, ts_y = torch.tensor(ts_x).float(), torch.tensor(ts_y).float()
	tr_x, tr_y = torch.tensor(tr_x).float(), torch.tensor(tr_y).float()
	val_x, val_y = tr_x[-n_tr:, :], tr_y[-n_tr:, :]
	tr_x, tr_y = tr_x[:n_tr, :], tr_y[:n_tr, :]

	network = {
		'i': 101,
		'h': 512,
		'o': 1,
		'l': 3
	}
	d_tr = (tr_x, tr_y)
	d_val = (val_x, val_y)
	d_ts = (ts_x, ts_y)
	if is_meta:
		loss_fn = nn.MSELoss()
		model = UnsupNet(network)
		rep_optim, aux_optim, prim_optim = model.get_optims(lr, wd)
		all_optim = AdamW(model.parameters(), lr=0, weight_decay=0)
		optims = rep_optim, prim_optim, aux_optim, all_optim
		dev_r2s, test_r2s, aux_y = make_n_run_meta(
													model, d_tr, d_val, aux, d_ts,
													n_epochs, phase_, loss_fn, optims,
													writer, id_='main'
												)

	else:
		dev_r2s, test_r2s, aux_y = make_n_run(
												network, d_tr, d_val, aux, d_ts,
												wd, lr, epochs=n_epochs
											)

	best_idx = np.argmax(dev_r2s)
	best_r2 = test_r2s[best_idx]
	print('		This is the R^2 = {:.5f} at epoch {}'.format(best_r2, best_idx))
	print('		Performing Self Training')

	# we can either use a classic model or a new one
	tr_x, tr_y = torch.vstack((tr_x, aux)), torch.vstack((tr_y, aux_y))
	d_tr = (tr_x, tr_y)

	if is_meta:
		loss_fn = nn.MSELoss()
		model = UnsupNet(network)
		rep_optim, aux_optim, prim_optim = model.get_optims(lr, wd)
		all_optim = AdamW(model.parameters(), lr=0, weight_decay=0)
		optims = rep_optim, prim_optim, aux_optim, all_optim
		dev_r2s, test_r2s, aux_y = make_n_run_meta(
													model, d_tr, d_val, aux, d_ts,
													n_epochs, phase_, loss_fn, optims,
													writer, id_='selftrain'
												)

	else:
		dev_r2s, test_r2s, aux_y = make_n_run(
												network, d_tr, d_val, aux, d_ts,
												wd, lr, epochs=int(n_epochs * 1.5)
											)

	best_idx = np.argmax(dev_r2s)
	best_r2 = test_r2s[best_idx]
	print('		This is the R^2 = {:.5f} at epoch {}'.format(best_r2, best_idx))


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
optimal_perf = r2_score(xy_ts[1], np.matmul(xy_ts[0][:, :problem['d']], w_opt.T))
print('This is the bayes optimal perf : ', optimal_perf)

print('Training Linear Model')
for alpha in [0, 0.1, 1, 10, 20, 30]:
	set_seed(0)
	clf = Ridge(alpha=alpha, fit_intercept=False).fit(xy_tr[0], xy_tr[1])
	score = r2_score(xy_ts[1], clf.predict(xy_ts[0]))
	print('This is when alpha = ', alpha)
	print('This is the R^2 = {:.5f} for naive training'.format(score))
	aux_pred = clf.predict(x_aux)
	new_x, new_y = np.vstack((xy_tr[0], x_aux)), np.vstack((xy_tr[1], aux_pred))


print('Training 3 Layer NN Model')
for lr in [1e-3, 7e-3, 1e-2, 3e-2]:
	for wd in [0, 1, 0.1, 0.01]:
		print('This is when lr = ', lr, ' wd = ', wd)
		# try:
		set_seed(0)
		writer = SummaryWriter('runs/no_labels.longest.Unsup_lr={:.3f}.wd={:.3f}'.format(lr, wd))
		train_nn(xy_tr[0], xy_tr[1], xy_ts[0], xy_ts[1], x_aux, lr=lr, n_epochs=200, wd=wd, is_meta=True, writer=writer)
		writer.flush()
		writer.close()
		# exit()
		# except:
		# 	print('		Issue arose')
