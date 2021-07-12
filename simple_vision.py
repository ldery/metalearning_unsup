
from argparse import ArgumentParser
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torchvision
from scipy.stats import bernoulli
from models.models import *


# Setup the arguments
def get_options():
	parser = ArgumentParser(description='Simple Test Bed For Meta-Learning Loss Functions')
	parser.add_argument(
							'-prim-split', type=float, default=0.8,
							help='What fraction of primary task data to use as train. Rest is used as val'
						)
	parser.add_argument('-num-epochs', type=int, default=100)
	parser.add_argument('-patience', type=int, default=11)
	parser.add_argument('-num_classes', type=int, default=10, help='number of classes in this dataset')
	parser.add_argument('-aux-frac', type=float, default=0.8, help='What fraction of auxiliary data to use')
	parser.add_argument('-aux-noise', type=float, default=0.5, help='Probability to flip the label of the auxiliary task')
	parser.add_argument(
							'-train-config', type=str,
							choices=['no_aux', 'self_train', 'add_aux', 'aux_head', 'meta_w_labels', 'unsup_meta']
						)
	parser.add_argument('-lr', type=float, default=1e-3)

	add_model_opts(parser)
	opts = parser.parse_args()
	return opts


def get_data(args):
	tform = torchvision.transforms.Compose([
						torchvision.transforms.ToTensor(),
					])
	# Todo [ldery] - will change this once I start testing out more datasets
	train_data = torchvision.datasets.MNIST('./', train=True, download=True, transform=tform)
	test_data = torchvision.datasets.MNIST('./', train=False, download=True, transform=tform)
	test_ = [[], []]
	for x, y in test_data:
		test_[0].append(x)
		test_[1].append(y)
	# Get the data, randomize and split
	shuffled_idxs = np.random.permutation(len(train_data))
	aux_end_idx = len(train_data) * int(args.aux_frac)
	aux_ids, shuffled_idxs = set(shuffled_idxs[:aux_end_idx]), shuffled_idxs[aux_end_idx:]
	prim_end_idx = len(shuffled_idxs) * int(args.prim_split)
	train_ids, val_ids = set(shuffled_idxs[:prim_end_idx]), set(shuffled_idxs[prim_end_idx:])
	aux_, train_, val_ = [[], []], [[], []], [[], []]
	X = bernoulli(args.aux_noise)
	for idx, (x, y) in enumerate(train_data):
		if idx in aux_ids:
			aux_[0].append(x)
			# Need to corrrupt the label according to a reasonable fraction
			if X.rvs(1)[0]:
				aux_[1].append(np.random.choice(args.num_classes, 1)[0])
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
	return aux_, train_, val_, test_


def batch_stats(model, xs, ys, loss_tensor=False):
	pred = model(xs)
	loss_tensor = model.loss_fn(pred, ys)
	loss = loss_tensor.item()
	pred = pred.argmax(dim=-1)
	correct_ = ys.eq(pred).sum().item()
	len_ = len(ys)
	loss_ = loss * len(ys)
	if not loss_tensor:
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
	return total_correct / total, total_loss / total


def relabel_aux(aux_, model, bsz=32):
	itr_ = get_iterator(data, batch_sz, shuffle=False)
	labels = []
	for xs, ys in itr_:
		pred = model(xs).argmax(dim=-1).cpu().numpy()
		labels.extend(pred)

	assert len(aux_[0]) == len(labels), 'Insufficient number of auxiliary data labels generated'
	aux_[1] = labels


def get_optim(args, model):
	optim = Adam(model.parameters(), lr=args.lr)
	scheduler = ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=5, min_lr=5e-6)
	return optim, scheduler


def get_iterator(data, bsz, shuffle=True):
	all_idxs = np.arange(len(data))
	if shuffle:
		all_idxs = np.random.permutation(len(data))
	while len(all_idxs) > 0:
		this_idxs = all_idxs[:bsz]
		all_idxs = all_idxs[bsz:]
		xs, ys = [], []
		for idx in this_idxs:
			xs.append(data[0][idx])
			ys.append(data[1][idx])
		xs, ys = torch.tensor(xs).float().cuda(), torch.tensor(ys).cuda()
		yield xs, ys


def train(train_, val_, test_, args):
	model = WideResnet({'main': args.num_classes}, args.depth, args.widen_factor, dropRate=args.dropRate)
	optim, scheduler = get_optim(args, model)
	model.train()
	val_accs, test_accs = [], []
	for i in range(args.num_epochs):
		itr_ = get_iterator(train_, batch_sz, shuffle=True)
		total_correct, total, total_loss = 0.0, 0.0, 0
		for xs, ys in itr_:
			correct_, loss_, len_, loss_tensor = batch_stats(model, xs, ys, loss_tensor=True)
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
		print("Epoch {} | Tr Loss = {:.3f} | Tr Acc = {:.3f}")
		print("Epoch {} | Vl Loss = {:.3f} | Vl Acc = {:.3f}")
		val_accs.append(val_avg_acc)
		test_accs.append(test_avg_acc)
		if scheduler is not None:
			scheduler.step(val_avg_acc)
		best_is_old = (np.argmax(val_accs) > i - args.patience)
		if (len(val_accs) > args.patience + 1) and best_is_old:
			break

	best_test = test_accs[np.argmax(val_accs)]
	return model, best_test


def train_model(args, all_data):
	aux_, train_, val_, test_ = all_data
	if args.train_config == 'no_aux':
		model, best_test = train(train_, val_, test_, args)
	elif args.train_config == 'add_aux':
		# join the train and aux data together
		train_[0].extend(aux_[0])
		train_[1].extend(aux_[1])
		model, best_test = train(train_, val_, test_, args)
	elif args.train_config == 'self_train':
		first_model, _ = train(train_, val_, test_, args)
		aux_ = relabel_aux(aux_, model)
		train_[0].extend(aux_[0])
		train_[1].extend(aux_[1])
		model, best_test = train(train_, val_, test_, args)
	# elif args.train_config == 'aux_head':
	# elif args.train_config == 'meta_w_labels':
	# elif args.train_config == 'unsup_meta':
	return best_test


if __name__ == "__main__":
	args = get_options()
	all_data = get_data(args)
	perf = train_model(args, all_data)
	print('Performance of {} is  Acc  = {:.3f}. Loss = {:.3f}'.format(args.train_config, perf[0], perf[1])


'''
Here is a setup
1. MNIST but with a fraction of the data
2. Some fraction of auxiliary data as MNIST with noisy labels
Here is a list of things to test out:
1. Perf without auxiliary data
2. Perf with auxiliary data + self training
3. Perf with auxiliary data + direct training with noisy labels
4. Perf with auxiliary data + training of noisy labels with a different head
5. Perf with auxiliary data + training of noisy labels witth diff head that meta-learns how to use noisy labels
6. Perf with auxiliary data + trainining of noisy labelled data in an unsupervised way
7. Semi-supervised options like VAT
I suspect that there will be a relationship between the efficacy and the degree of noise / corruptoin. 
'''

