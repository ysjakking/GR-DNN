import gzip
import six.moves.cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image

print('... loading data')

model_version_name='model1'

dataset='./mnist.pkl.gz'
code_name_DAE='./test_codes_data_%s.mat'% (model_version_name)
code_name_GRDNN='./test_codes_GRDNN_%s.mat'% (model_version_name)

# Load the dataset
with gzip.open(dataset, 'rb') as f:
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    except:
        train_set, valid_set, test_set = pickle.load(f)

data_x, data_y = test_set

mat_contents_DAE = sio.loadmat(code_name_DAE)
mat_contents_GRDNN = sio.loadmat(code_name_GRDNN)

code_DAE = mat_contents_DAE['test_codes']
code_GRDNN = mat_contents_GRDNN['test_codes']

####################view 2D embedding
#code_DAE
colors=['red', 'green', 'blue', 'brown','cyan','magenta','yellow','black','coral','steelblue']
fig, ax = plt.subplots()
for temp_label in range(10):
    index=np.where(data_y==temp_label)
    x = code_DAE[index[0],:]
    ax.scatter(x[:,0], x[:,1],s=2, c=colors[temp_label], label=temp_label,alpha=1, edgecolors='none')


ax.legend()
plt.savefig('./result/DAE_%s.jpg'% (model_version_name))



#code_GRDNN
fig, ax = plt.subplots()
for temp_label in range(10):
    index=np.where(data_y==temp_label)
    x = code_GRDNN[index[0],:]
    ax.scatter(x[:,0], x[:,1],s=2, c=colors[temp_label], label=temp_label,alpha=1, edgecolors='none')

ax.legend()
plt.savefig('./result/GRDNN_%s.jpg'%(model_version_name))



####################generate and save reconstructions
recons_DAE = mat_contents_DAE['test_reconstruct']
recons_GRDNN = mat_contents_GRDNN['test_reconstruct']

samples=20 #for the first 20 image samples
dims=recons_DAE.shape[1]
dims=int(np.sqrt(dims))

original=data_x[0:samples,:]
recons_DAE=recons_DAE[0:samples,:]
recons_GRDNN=recons_GRDNN[0:samples,:]

original=np.reshape(original,[samples*dims,dims])
recons_DAE1=np.reshape(recons_DAE,[samples*dims,dims])
recons_GRDNN1=np.reshape(recons_GRDNN,[samples*dims,dims])

recons_DAE1=np.concatenate([original,recons_DAE1,recons_GRDNN1],1)

result1 = Image.fromarray((recons_DAE1* 255).astype(np.int8))
result1 = result1.convert('RGB')
result1.save('./result/reconstruct_%s.jpg'%(model_version_name))
