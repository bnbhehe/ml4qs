from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pprint import pprint
import numpy as np
import json
import sklearn.preprocessing
import time
import argparse
import os.path
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE 
from matplotlib import pyplot as plt
import itertools

def plot_histograms(X,title='x measurements',bins=100):
	rcParams['figure.figsize'] = (10,10)
	signal = X
	hist,bin_edges = np.histogram(signal,bins = bins,normed=1)
	width = 0.9 * (bin_edges[1] - bin_edges[0])
	center = (bin_edges[:-1] + bin_edges[1:]) /2

	mu = np.mean(signal)
	sigma = np.std(signal)
	y = mlab.normpdf(bin_edges, mu, sigma)
	plt.bar(center,hist,align='center',width=width)
	plt.plot(bin_edges, y, 'r--', linewidth=3)
	plt.title('%s'%(title + " bins=" + str(bins)))
	plt.savefig(title+'.png')




def save_conf_matrix(y_test,preds,out,path,normalize_conf=False,classes = np.arange(10)):
	f = open(os.path.join(path,'accuracy.txt'),'w+')
	f.write(out)
	f.flush()
	f.close()
	conf_matrix = confusion_matrix(np.argmax(y_test,axis=1),np.argmax(preds,axis=1))
	_plot_confusion_matrix(conf_matrix,classes=classes,normalize=normalize_conf,title='Unormalized confusion matrix',path=path)



def visualize(features,labels,filename = 'tsne.png',annotate=False,plot_only = 1000,load=False,classes=np.arange(10)):
	assert features.shape[0] >= len(labels), "More labels than weights"
	print('Plotting features of dimensionality:%d'%features.shape[1])
	print('Saving to filename: %s'%filename)
	projections_file = 'features_2d.npy'
	tsne = TSNE(perplexity=30, n_components=2, n_iter=5000,verbose=3,init='random')
	projection_2d = tsne.fit_transform(features)
	print('fitting features complete...')
	np.save(projections_file,projection_2d)
	plt.figure(figsize=(10, 10))  #in inches
	xx = np.arange(len(classes))
	tt = np.random.randn(len(classes))
	print('plotting features ...',projection_2d.shape)
	for i in range(xx.shape[0]):
		proxy = plt.scatter(xx[i],tt[i],
			color=colors[i],
			marker=markers[i],s=10,
			label=classes[i])
	print(labels)
	print(projection_2d.shape)
	for i,label in enumerate(labels):
		x = projection_2d[i,0]
		y = projection_2d[i,1]
		plt.scatter(x, y ,color = colors[label],s=30,marker=markers[label])
		if annotate:
			plt.annotate(classes[label],xy=(x, y),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom')
			
			
			
def _visualize_regrr(features,labels,filename = 'tsne.png',plot_only = 1000,load=False):
	assert features.shape[0] >= len(labels), "More labels than weights"
	print('Plotting features of dimensionality:%d'%features.shape[1])
	tsne = TSNE(perplexity=30, n_components=2, n_iter=5000,verbose=3,init='random')
	projection_2d = tsne.fit_transform(features)
	print('fitting features complete...')
	plt.figure(figsize=(10, 10))  #in inches
	cm = plt.cm.get_cmap('RdYlBu')(np.linspace(np.min(labels),np.max(labels),100))
	
	x = projection_2d[:,0]
	y = projection_2d[:,1]
	sc = plt.scatter(x, y, c=labels, vmin=0, vmax=20, s=35, cmap=cm)
	plt.colorbar(sc)
	plt.show()

def _plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues,path = './default/'):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	print('Creating confusion matrix...')
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



  	thresh = cm.max() / 2.
  	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	  	plt.text(j, i, cm[i, j],
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

  	plt.ylabel('True label')
  	plt.xlabel('Predicted label')
  	if not os.path.exists(path):
		os.makedirs(path)

  	print('Writing confusion matrix to path %s'%(path))     
  	plt.savefig(path+'_conf_matrix.png')
  	plt.show()
