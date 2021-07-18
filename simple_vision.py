
from argparse import ArgumentParser
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torchvision
from scipy.stats import bernoulli
from models.models import *
import pdb
import random
import higher
import gc
from copy import deepcopy


# Setup the arguments
def get_options():
	parser = ArgumentParser(description='Simple Test Bed For Meta-Learning Loss Functions')
	parser.add_argument(
							'-prim-split', type=float, default=0.8,
							help='What fraction of primary task data to use as train. Rest is used as val'
						)
	parser.add_argument('-seed', type=int, default=0)
	parser.add_argument('-num-epochs', type=int, default=100)
	parser.add_argument('-patience', type=int, default=11)
	parser.add_argument('-num_classes', type=int, default=10, help='number of classes in this dataset')
	parser.add_argument('-aux-frac', type=float, default=0.8, help='What fraction of auxiliary data to use')
	parser.add_argument('-aux-noise', type=float, default=0.5, help='Probability to flip the label of the auxiliary task')
	parser.add_argument(
							'-train-config', type=str, default='no_aux',
							choices=['no_aux', 'add_aux', 'aux_head', 'meta_w_labels', 'unsup_meta']
						)
	parser.add_argument('-self-train', action='store_true')
	parser.add_argument('-lr', type=float, default=1e-3)
	parser.add_argument('-bsz', type=int, default=64)
	parser.add_argument('-aux_bsz', type=int, default=128)
	parser.add_argument('-iterated-steps', type=int, default=1)
	parser.add_argument('-aux-wd', type=float, default=1e-2)
	parser.add_argument('-dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'])
	parser.add_argument('-img_channels', type=int, default=3)

	add_model_opts(parser)
	opts = parser.parse_args()
	return opts


def get_data(args):
	tform = torchvision.transforms.Compose([
						torchvision.transforms.ToTensor(),
						torchvision.transforms.Resize(32),
					])
	# Todo [ldery] - will change this once I start testing out more datasets
	if args.dataset == 'MNIST':
		train_data = torchvision.datasets.MNIST('.', train=True, download=True, transform=tform)
		test_data  = torchvision.datasets.MNIST('.', train=False, download=True, transform=tform)
	elif args.dataset == 'CIFAR10':
		train_data = torchvision.datasets.CIFAR10('.', train=True, download=True, transform=tform)
		test_data  = torchvision.datasets.CIFAR10('.', train=False, download=True, transform=tform)
	else:
		raise ValueError('Incorrect dataset options : {}'.format(args.dataset))
	test_ = [[], []]
	for x, y in test_data:
		test_[0].append(x)
		test_[1].append(y)
	# Get the data, randomize and split
	shuffled_idxs = np.random.permutation(len(train_data))
	aux_end_idx = int(len(train_data) * args.aux_frac)
	aux_ids, shuffled_idxs = set(shuffled_idxs[:aux_end_idx]), shuffled_idxs[aux_end_idx:]
	prim_end_idx = int(len(shuffled_idxs) * args.prim_split)
	train_ids, val_ids = set(shuffled_idxs[:prim_end_idx]), set(shuffled_idxs[prim_end_idx:])
	aux_, train_, val_ = [[], []], [[], []], [[], []]
	X = bernoulli(args.aux_noise)
	for idx, (x, y) in enumerate(train_data):
		if idx in aux_ids:
			aux_[0].append(x)
			# Need to corrrupt the label according to a reasonable fraction
			if X.rvs(1)[0]:
				choices = list(set(range(args.num_classes)) - set([y]))
				aux_[1].append(np.random.choice(choices, 1)[0])
			else:
				aux_[1].append(y)
		elif idx in train_ids:
			train_[0].append(x)
			train_[1].append(y)
		elif idx in val_ids:
			val_[0].append(x)
			val_[1].append(y)
		else:
			raise ValueError("This idx {} is not in any split".format(idx))
	print('There are {} training points'.format(len(train_[0])))
	return aux_, train_, val_, test_


def batch_stats(model, xs, ys, head_name='main', return_loss=False):
	pred = model(xs, head_name=head_name)
	if 'meta' in head_name:
		loss_tensor = pred
		return loss_tensor
	else:
		loss_tensor = model.loss_fn(pred, ys)
	loss = loss_tensor.item()
	pred = pred.argmax(dim=-1)
	correct_ = ys.eq(pred).sum().item()
	len_ = len(ys)
	loss_ = loss * len(ys)
	if not return_loss:
		return correct_, loss_, len_
	else:
		return correct_, loss_, len_, loss_tensor


def evaluate(model, data, bsz=32):
	itr_ = get_iterator(data, bsz)
	model.eval()
	total_correct, total, total_loss = 0.0, 0.0, 0
	for xs, ys in itr_:
		correct_, loss_, len_ = batch_stats(model, xs, ys)
		total_correct += correct_
		total_loss += loss_
		total += len_
	model.train()
	return total_correct / total, total_loss / total


def relabel_aux(aux_, model, head_name='main', bsz=32):
	itr_ = get_iterator(aux_, bsz, shuffle=False)
	labels = []
	for xs, ys in itr_:
		pred = model(xs, head_name=head_name).argmax(dim=-1).cpu().numpy()
		labels.extend(pred)

	assert len(aux_[0]) == len(labels), 'Insufficient number of auxiliary data labels generated'
	aux_[1] = labels
	return aux_


def get_optim(args, model):
	optim = Adam(model.parameters(), lr=args.lr)
	scheduler = ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=5, min_lr=5e-6)
	return optim, scheduler

def get_optims_for_meta(args, model):
	optim = Adam(model.parameters(), lr=args.lr)
	scheduler = ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=5, min_lr=5e-6)
	return optim, scheduler

def get_iterator(data, bsz, shuffle=True):
	all_idxs = np.arange(len(data[0]))
	if shuffle:
		all_idxs = np.random.permutation(len(data[0]))
	while len(all_idxs) > 0:
		this_idxs = all_idxs[:bsz]
		all_idxs = all_idxs[bsz:]
		xs, ys = [], []
		for idx in this_idxs:
			xs.append(data[0][idx])
			ys.append(data[1][idx])
		xs, ys = torch.stack(xs).float().cuda(), torch.tensor(ys).cuda()
		yield xs, ys


def train(train_, val_, test_, args, aux_=None):
	out_dict = {'main': args.num_classes}
	# Include the auxiliary task here
	aux_present = aux_ is not None
	if aux_present:
		out_dict['aux'] = args.num_classes
	model = WideResnet(out_dict, args.depth, args.widen_factor, insize=args.img_channels, dropRate=args.dropRate)
	model.cuda()
	optim, scheduler = get_optim(args, model)
	val_accs, test_accs, val_losses = [], [], []
	best_idx, best_model = -1, None
	for i in range(args.num_epochs):
		itr_ = get_iterator(train_, args.bsz, shuffle=True)
		if aux_present:
			aux_itr_ = get_iterator(aux_, args.bsz, shuffle=True)
		total_correct, total, total_loss = 0.0, 0.0, 0
		for xs, ys in itr_:
			correct_, loss_, len_, loss_tensor = batch_stats(model, xs, ys, return_loss=True)
			if aux_present:
				aux_xs, aux_ys = next(aux_itr_)
				_, _, _, aux_loss = batch_stats(model, aux_xs, aux_ys, head_name='aux', return_loss=True)
				aux_loss.backward()
			total_correct += correct_
			total_loss += loss_
			total += len_
			loss_tensor.backward()
			optim.step()
			optim.zero_grad()
		# Do the training
		train_avg_acc, train_avg_loss = total_correct / total, total_loss / total
		val_avg_acc, val_avg_loss = evaluate(model, val_)
		test_avg_acc, test_avg_loss = evaluate(model, test_)
		print("Epoch {} | Tr Loss = {:.3f} | Tr Acc = {:.3f}".format(i, train_avg_loss, train_avg_acc))
		print("         | Vl Loss = {:.3f} | Vl Acc = {:.3f}".format(val_avg_loss, val_avg_acc))
		print("         | Ts Loss = {:.3f} | Ts Acc = {:.3f}".format(test_avg_loss, test_avg_acc))
		print('\n')
		val_accs.append(val_avg_acc)
		test_accs.append(test_avg_acc)
		val_losses.append(val_avg_loss)
		if scheduler is not None:
			scheduler.step(val_avg_acc)
		# Save the best model
		if val_avg_acc >= np.argmax(val_accs):
			best_idx = i
			best_model = deepcopy(model)
		best_is_old = (best_idx < i - args.patience)
		if (len(val_accs) > args.patience + 1) and best_is_old:
			break

	best_test = test_accs[best_idx]
	print('Best index is {}. And Acc {}'.format(best_idx, best_test))
	return best_model, best_test

def get_aux_start_end(model):
	auxparam_start, auxparam_end = -1, -1
	for idx, (name, _) in enumerate(model.named_parameters()):
		if 'meta_aux' in name:
			if auxparam_start == -1:
				auxparam_start = idx
			auxparam_end = idx + 1
	assert auxparam_start > -1 and auxparam_end > -1, 'Start and End idxs for aux not set'
	return auxparam_start, auxparam_end

def get_meta_optims(args, model):
	all_optim = Adam(model.parameters(), lr=args.lr)

	# bad design but don't really care atm
	primhead = getattr(model.model, "fc-main", None)
	assert primhead is not None, 'The primary head is not intialized'
	primhead_optim = Adam(primhead.parameters(), lr=args.lr)
	primhead_scheduler = ReduceLROnPlateau(primhead_optim, mode='max', factor=0.5, patience=5, min_lr=5e-6)

	aux_optim = AdamW(model.model.meta_auxmodel.parameters(), lr=args.lr, weight_decay=args.aux_wd)
	auxhead_scheduler = ReduceLROnPlateau(aux_optim, mode='max', factor=0.5, patience=5, min_lr=5e-6)
	
	body_params = []
	for name, p in model.named_parameters():
		if 'fc' not in name:
			body_params.append(p)
	body_optim = Adam(body_params, lr=args.lr)
	body_scheduler = ReduceLROnPlateau(body_optim, mode='max', factor=0.5, patience=5, min_lr=5e-6)

	return (primhead_optim, body_optim, aux_optim, all_optim), \
			(primhead_scheduler, auxhead_scheduler, body_scheduler)


def meta_train(train_, val_, test_, aux_, args):
	out_dict = {'main': args.num_classes}
	model = WideResnet(out_dict, args.depth, args.widen_factor, insize=args.img_channels, dropRate=args.dropRate)
	model.cuda()
	model.add_heads({'meta_aux': 1})
# 	model.add_heads({'proxy_main': args.num_classes})
	optims, schedulers = get_meta_optims(args, model)
	prim_head_optim, body_optim, aux_optim, all_optim = optims
	auxparam_start, auxparam_end = get_aux_start_end(model)

	val_accs, test_accs, val_losses = [], [], []
	best_idx, best_model = -1, None
	for i in range(args.num_epochs):
		# For iterated_steps, keep the body and aux-model fixed and learn a primary task head
		# reset the head so we have a fresh embedding : 
# 		model.reset_head('main')
		torch.cuda.empty_cache()
		gc.collect()
		for j in range(args.iterated_steps):
			itr_ = get_iterator(train_, args.bsz, shuffle=True)
			for xs, ys in itr_:
				_, _, _, loss_tensor = batch_stats(model, xs, ys, return_loss=True)
				loss_tensor.backward()
				prim_head_optim.step()
				all_optim.zero_grad()
		
		# For n iterates, keep train aux model and keep rest fixed
		for j in range(args.iterated_steps):
			itr_ = get_iterator(val_, args.bsz, shuffle=True)
			aux_itr_ = get_iterator(aux_, args.aux_bsz, shuffle=True)
			for xs, ys in itr_:
				aux_x, _ = next(aux_itr_)
				with higher.innerloop_ctx(model, body_optim, track_higher_grads=True) as (fmodel, diff_bodyoptim):
					this_auxloss = fmodel(aux_x, head_name='meta_aux')
					diff_bodyoptim.step(this_auxloss)

					prim_loss = fmodel.loss_fn(fmodel(xs, head_name='main'), ys)
					aux_params = list(fmodel.parameters(time=0))[auxparam_start:auxparam_end]
					aux_grads = torch.autograd.grad(prim_loss, aux_params, allow_unused=True)
					for p, g in zip(model.model.meta_auxmodel.parameters(), aux_grads):
						if p.grad is None:
							p.grad = torch.zeros_like(p)
						p.grad.add_(g)

					aux_optim.step()
					aux_optim.zero_grad()

		# For n iterates, keep train body and keep rest fixed
		total_correct, total, total_loss = 0.0, 0.0, 0.0
		aux_total, total_auxloss = 0.0, 0.0
		for j in range(args.iterated_steps):
			aux_itr_ = get_iterator(aux_, args.aux_bsz, shuffle=True)
			for aux_xs, aux_ys in aux_itr_:
				# get on auxiliary
# 				aux_xs, aux_ys = next(aux_itr_)
				aux_loss = batch_stats(model, aux_xs, aux_ys, head_name='meta_aux', return_loss=True)
				aux_loss.backward()
				
				body_optim.step()
				all_optim.zero_grad()
				
				total_auxloss += (aux_loss.item())*len(aux_ys)
				aux_total += len(aux_ys)
			
			itr_ = get_iterator(train_, args.bsz, shuffle=True)
			for xs, ys in itr_:
				# get on primary
				correct_, loss_, len_, loss_tensor = batch_stats(model, xs, ys, return_loss=True)
				loss_tensor.backward()
				
				
				body_optim.step()
				all_optim.zero_grad()

				total_correct += correct_
				total_loss += loss_
				total += len_

		train_avg_acc, train_avg_loss = total_correct / total, total_loss / total
		aux_avg_loss = total_auxloss / aux_total
		val_avg_acc, val_avg_loss = evaluate(model, val_)
		test_avg_acc, test_avg_loss = evaluate(model, test_)
		print("Epoch {} | Tr Loss = {:.3f} | Tr Acc = {:.3f} | Aux Loss = {:.3f}".format(i, train_avg_loss, train_avg_acc, aux_avg_loss))
		print("        | Vl Loss = {:.3f} | Vl Acc = {:.3f}".format(val_avg_loss, val_avg_acc))
		print("        | Ts Loss = {:.3f} | Ts Acc = {:.3f}".format(test_avg_loss, test_avg_acc))
		print('\n')
		val_accs.append(val_avg_acc)
		test_accs.append(test_avg_acc)
		val_losses.append(val_avg_loss)
# 		if schedulers is not None:
# 			for scheduler in schedulers:
# 				scheduler.step(val_avg_acc)
		# Save the best model
		if val_avg_acc >= np.argmax(val_accs):
			best_idx = i
			best_model = deepcopy(model)
		best_is_old = (best_idx < i - args.patience)
		if (len(val_accs) > args.patience + 1) and best_is_old:
			break

	best_test = test_accs[best_idx]
	print('Best index is {}. And Acc {}'.format(best_idx, best_test))
	return best_model, best_test


def train_model(args, all_data):
	aux_, train_, val_, test_ = all_data
	if args.train_config == 'no_aux':
		model, best_test = train(train_, val_, test_, args)
	elif args.train_config == 'add_aux':
		# join the train and aux data together
		train_[0].extend(aux_[0])
		train_[1].extend(aux_[1])
		model, best_test = train(train_, val_, test_, args)
	elif args.train_config == 'aux_head':
		model, best_test = train(train_, val_, test_, args, aux_=aux_)
	elif args.train_config == 'unsup_meta':
		model, best_test = meta_train(train_, val_, test_, aux_, args)
	else:
		raise ValueError('Wrong train config specfiication {}'.format(args.train_config))

	if args.self_train:
		print('Performing Self-Training')
		aux_ = relabel_aux(aux_, model)
		train_[0].extend(aux_[0])
		train_[1].extend(aux_[1])
		model, best_test = train(train_, val_, test_, args)
	return best_test


def set_random_seed(seed):
	# Esp important for ensuring deterministic behavior with CNNs
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	cuda_available = torch.cuda.is_available()
	if cuda_available:
		torch.cuda.manual_seed_all(seed)
	return cuda_available


if __name__ == "__main__":
	args = get_options()
	set_random_seed(args.seed)
	all_data = get_data(args)
	perf = train_model(args, all_data)
	print('Performance of {} is  Acc  = {:.3f}'.format(args.train_config, perf))

