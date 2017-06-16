### 1.2 Histogram
# YOUR CODE HERE
%pylab inline
import os
from sklearns.manifold import TSNE
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


def _visualize(features,labels,filename = 'tsne.png',annotate=False,plot_only = 1000,load=False):
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
