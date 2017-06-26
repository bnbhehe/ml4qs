
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
get_ipython().magic(u'pylab inline')
import pandas as pd
import json
import seaborn
import copy
import os
from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TextAbstraction import TextAbstraction
from util.VisualizeDataset import VisualizeDataset
from util import util
from util.parser import get_labels,generate_csv,generate_labels,get_sensor_values
import time


# # Below cell has global variables (path-names mostly)

# In[2]:

dataset_folder = './datasets'
files = os.listdir(dataset_folder)
print(files)
dataset_fname  = 'log_assignment3'
event_fname = 'log_events_assignment3'
result_dataset_path = './intermediate_datafiles/'
csv_dataset_path = os.path.join(dataset_folder,'csv/')
dataset_path = os.path.join(dataset_folder,dataset_fname)
event_path = os.path.join(dataset_folder,event_fname)
output_fname = 'dataset.csv'


# In[3]:

if not os.path.exists(result_dataset_path):
    print('Creating result directory: ' + result_dataset_path)
    os.makedirs(result_dataset_path)
if not os.path.exists(csv_dataset_path):
    print('Creating result directory: ' + csv_dataset_path)
    os.makedirs(csv_dataset_path)


# ### LOG PARSING FUNCTION CALL

# In[4]:

labels = get_labels(dataset_path)
print('Label Mappings:\n')
print(json.dumps(labels,indent=2))


# In[5]:

sensors_dict = get_sensor_values(dataset_path)
print('Sensors in log file')
sensors = sensors_dict.keys()
print(json.dumps(sensors,indent=2))


# # CSV generation cell

# In[ ]:

for sensor in sensors:
    if sensor == 'rotation vector':
        generate_csv(sensors_dict[sensor],
                     header=['x','y','z','theta','phi','timestamp'],
                     fname = os.path.join(csv_dataset_path,sensor+'.csv'))
    else:
        generate_csv(sensors_dict[sensor],fname = os.path.join(csv_dataset_path,sensor+'.csv'))
generate_labels(fname=event_path,log_fname=dataset_path,fout=csv_dataset_path+'labels.csv');


# # Dataset creation cell

# In[ ]:

granularities = [250,1000,10000]
datasets = []


for milliseconds_per_instance in granularities:
    initial = time.time()
    # Create an initial dataset object with the base directory for our data and a granularity
    DataSet = CreateDataset(csv_dataset_path, milliseconds_per_instance)
    print('granularity is %d ms'%milliseconds_per_instance)
    
    DataSet.add_numerical_dataset('bmi160 accelerometer.csv','timestamp',['x','y','z'],'avg','acc_phone_')
    DataSet.add_numerical_dataset('bmi160 gyroscope.csv','timestamp',['x','y','z'],'avg','gyr_phone_')
    DataSet.add_numerical_dataset('rotation vector.csv','timestamp',['x','y','z','theta','phi'],'avg','rotation_phone_')
    DataSet.add_event_dataset('labels.csv','label_start','label_end','label','binary')
    print('Added all sensors and labels')
    dataset = DataSet.data_table
#     util.print_statistics(dataset)
    datasets.append(copy.deepcopy(dataset))
    dataset.to_csv(result_dataset_path + '/'+str(milliseconds_per_instance)+ '_dataset.csv')
    print('Dataset created,\tDuration:%.5f seconds'%(time.time()-initial))
    print('====================================')
    


# # Some visualizations

# In[ ]:


# for i in range(len(datasets)):
#     DataViz.plot_dataset(datasets[i],['acc_phone'],['like'],['line'])
#     DataViz.plot_dataset(datasets[i],['gyr_phone'],['like'],['line'])
#     DataViz.plot_dataset(datasets[i],['rotation_phone'],['like'],['line'])
# #     DataViz.plot_dataset(datasets[i],['label'],['like'],['points','points'])


# # Data Filtering Cell (U can continue from here)

# In[6]:

### choose a dataset to filter:
DataViz = VisualizeDataset()
rcParams['figure.figsize'] = (20,10)
data_index = 0
try:
    dataset = pd.read_csv(os.path.join(result_dataset_path,datasets[data_index])
                          ,index_col=0) # minimum granularity - biggest dataset
    millis = granularities[data_index]
except NameError:
    print(os.listdir(result_dataset_path))
    datasets = os.listdir(result_dataset_path)
    datasets = [x for x in datasets if 'dataset' in x ]
    print(datasets)
    granularities = [int(x.split('_')[0]) for x in datasets if 'dataset' in x]
    print(datasets,granularities)
    dataset = pd.read_csv(os.path.join(result_dataset_path,datasets[data_index])
                          ,index_col=0) # minimum granularity - biggest dataset
    millis = granularities[data_index]
    
dataset.index = pd.to_datetime(dataset.index)
milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000
print("Using milliseconds per instance: %d"%milliseconds_per_instance)
outlier_columns = ['acc_phone_x', 'gyr_phone_y']
dataset.head()


# In[7]:

OutlierDistr = DistributionBasedOutlierDetection()
OutlierDist = DistanceBasedOutlierDetection()

#And investigate the approaches for all relevant attributes.
for col in outlier_columns:
    # And try out all different approaches. Note that we have done some optimization
    # of the parameter values for each of the approaches by visual inspection.
    dataset = OutlierDistr.chauvenet(dataset, col)
#     DataViz.plot_binary_outliers(dataset, col, col + '_outlier')
    dataset = OutlierDistr.mixture_model(dataset, col)
#     DataViz.plot_dataset(dataset, [col, col + '_mixture'], ['like','like'], ['line', 'points'])
    # This requires:
    # n_data_points * n_data_points * point_size =
    # 31839 * 31839 * 64 bits = ~8GB available memory
    try:
        dataset = OutlierDist.simple_distance_based(dataset, [col], 'euclidean', 0.10, 0.99)
#         DataViz.plot_binary_outliers(dataset, col, 'simple_dist_outlier')
    except MemoryError as e:
        print('Not enough memory available for simple distance-based outlier detection...')
        print('Skipping.')
    
    try:
        dataset = OutlierDist.local_outlier_factor(dataset, [col], 'euclidean', 5)
#         DataViz.plot_dataset(dataset, [col, 'lof'], ['exact','exact'], ['line', 'points'])
    except MemoryError as e:
        print('Not enough memory available for lof...')
        print('Skipping.')

    # Remove all the stuff from the dataset again.
    cols_to_remove = [col + '_outlier', col + '_mixture', 'simple_dist_outlier', 'lof']
    for to_remove in cols_to_remove:
        if to_remove in dataset:
            del dataset[to_remove]

# We take Chauvent's criterion and apply it to all but the label data...

for col in [c for c in dataset.columns if not 'label' in c]:
#     print('Measurement is now: ' , col)
    dataset = OutlierDistr.chauvenet(dataset, col)
    dataset.loc[dataset[col + '_outlier'] == True, col] = np.nan
    del dataset[col + '_outlier']
dataset.to_csv(result_dataset_path +str(millis)+'_outlier.csv')


# ### Apply Kalman Filtering to the dataset

# In[81]:

rcParams['figure.figsize'] = (20,10)
missing_values = ['acc_phone_x','acc_phone_y','acc_phone_z','gyr_phone_x','gyr_phone_y','gyr_phone_z','rotation_phone_x','rotation_phone_y']

original_dataset = pd.read_csv(result_dataset_path+str(millis)+'_outlier.csv',index_col=0)
original_dataset.index = pd.to_datetime(original_dataset.index)
KalFilter = KalmanFilters()
value = missing_values[0]
print('Filtering out %s value'%value)
kalman_dataset = KalFilter.apply_kalman_filter(original_dataset,value)
DataViz.plot_dataset(kalman_dataset, [value,value+'_kalman'], ['exact','exact'], ['line', 'line'])
for value in missing_values[1:]:
    print('Filtering out %s value'%value)
    kalman_dataset = KalFilter.apply_kalman_filter(kalman_dataset,value)
#     DataViz.plot_dataset(kalman_dataset, [value,value+'_kalman'], ['exact','exact'], ['line', 'line'])
kalman_dataset.to_csv(result_dataset_path+'/'+str(millis)+'_kalman_dataset.csv')


# # Apply PCA to get principal component values.

# In[82]:

PCA = PrincipalComponentAnalysis()
# selected_predictor_cols = [c for c in dataset.columns if (not ('label' in c)) and (not (c == 'rotation_phone_theta'))]
selected_predictor_cols = missing_values
selected_predictor_cols = [x+'_kalman' for x in selected_predictor_cols]
selected_predictor_cols += ['rotation_phone_z','rotation_phone_theta','rotation_phone_phi']
print(selected_predictor_cols)
pc_values = PCA.determine_pc_explained_variance(kalman_dataset, selected_predictor_cols)
plt.plot(range(1, len(selected_predictor_cols)+1), pc_values, 'b-')
plt.xlabel('principal component number')
plt.ylabel('explained variance')
plt.show()
#Derived from the plot
n_pcs = 6


# In[105]:


kalman_dataset = PCA.apply_pca(copy.deepcopy(kalman_dataset), selected_predictor_cols, n_pcs)
kalman_dataset.head()


# In[117]:

rcParams['figure.figsize'] = (20,3)
DataViz.plot_dataset(kalman_dataset, ['pca_'], ['like'], ['line'])
plt.plot(kalman_dataset['labelstairsup']+0.02,'ro')
plt.plot(kalman_dataset['labelstairsdown']-0.02,'bo')
plt.plot(kalman_dataset['labelwalking']+0.03,'go')
plt.legend(['stairs up','stairs down','walking'])
plt.show()


# In[122]:

kalman_dataset.to_csv(result_dataset_path+'/'+str(millis)+'_dataset_rest.csv')


# ### Feature Engineering

# In[41]:

### choose a dataset to filter:
DataViz = VisualizeDataset()
millis = 250
rcParams['figure.figsize'] = (20,10)
data_index = 0
dataset = pd.read_csv(os.path.join(result_dataset_path,str(millis)+'_dataset_rest.csv'),index_col=0)
dataset.index = pd.to_datetime(dataset.index)
milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000
print("Using milliseconds per instance: %d"%milliseconds_per_instance)
dataset.head()


# In[47]:

window_sizes = [int(float(5000)/milliseconds_per_instance), int(float(0.5*60000)/milliseconds_per_instance), int(float(2.*60000)/milliseconds_per_instance)]
print(window_sizes)
NumAbs = NumericalAbstraction()
CatAbs = CategoricalAbstraction()

dataset_copy = copy.deepcopy(dataset)
for ws in window_sizes:
    dataset_copy = NumAbs.abstract_numerical(dataset_copy, ['acc_phone_x_kalman'], ws, 'mean')
    dataset_copy = NumAbs.abstract_numerical(dataset_copy, ['acc_phone_x_kalman'], ws, 'std')
rcParams['figure.figsize'] = (20,2)
plt.plot(dataset_copy['acc_phone_x_kalman_temp_mean_ws_20'])
plt.plot(dataset_copy['acc_phone_x_kalman_temp_mean_ws_120'])
plt.plot(dataset_copy['acc_phone_x_kalman_temp_mean_ws_480'])
plt.title('acc x mean with window size of 20,120,480')
plt.show()
plt.plot(dataset_copy['acc_phone_x_kalman_temp_std_ws_20'])
plt.plot(dataset_copy['acc_phone_x_kalman_temp_std_ws_120'])
plt.plot(dataset_copy['acc_phone_x_kalman_temp_std_ws_480'])
plt.title('acc x std with window size of 20,120,480')
plt.show()


# In[48]:

# DataViz.plot_dataset(dataset_copy, ['acc_phone_x', 'acc_phone_x_temp_mean', 'acc_phone_x_temp_std'], ['exact', 'like', 'like'], ['line', 'line', 'line'])
# print(dataset_copy.keys())

ws = int(float(0.1*60000)/milliseconds_per_instance)
selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]
dataset_ws = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
dataset_ws = NumAbs.abstract_numerical(dataset_ws, selected_predictor_cols, ws, 'std')
dataset_ws = CatAbs.abstract_categorical(dataset_ws, ['label'], ['like'], 0.03, int(float(5*60000)/milliseconds_per_instance), 2)

dataset_ws.to_csv(result_dataset_path+'/'+'dataset_ws_'+str(ws)+'.csv')


# ### Frequency Domain - Fourier Transformations.

# In[50]:

FreqAbs = FourierTransformation()
fs = float(float(1000)/milliseconds_per_instance)
print("Frequency for FFT: %5.5f"%fs)
periodic_predictor_cols =['acc_phone_x_kalman', 'acc_phone_y_kalman',
                          'acc_phone_z_kalman', 'gyr_phone_x_kalman', 
                          'gyr_phone_y_kalman', 'gyr_phone_z_kalman', 
                          'rotation_phone_x_kalman', 'rotation_phone_y_kalman', 
                          'rotation_phone_z', 'rotation_phone_theta', 'rotation_phone_phi']
dataset_fft = FreqAbs.abstract_frequency(dataset_ws, periodic_predictor_cols, int(float(10000)/milliseconds_per_instance), fs)
for col in periodic_predictor_cols[1:]:
    dataset_fft = FreqAbs.abstract_frequency(dataset_ws, periodic_predictor_cols, int(float(10000)/milliseconds_per_instance), fs)
print("Finished with fft transformations...")
window_overlap = 0.9
skip_points = int((1-window_overlap) * ws)
dataset_fft = dataset_fft.iloc[::skip_points,:]


# In[53]:




# In[66]:

print(dataset_fft.keys(),'\nNumber of features',len(dataset_fft.keys()))


# In[65]:

rcParams['figure.figsize'] = (100,50)
DataViz.plot_dataset(dataset_fft,['acc_phone_x_kalman_freq_'],['like'],['line'])


# In[67]:

dataset_fft.to_csv(result_dataset_path+'/'+str(millis)+'_'+'final_dataset_fft_'+str(fs)+'_ws_'+str(ws)+'.csv')


# In[ ]:



