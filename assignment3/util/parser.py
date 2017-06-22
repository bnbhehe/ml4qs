import os
import json
import ast
from itertools import izip
import pandas as pd
def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return izip(a, a)

def get_labels(fname):
    label_mappings = {}
    with open(fname,'r+') as f:
        for line in f.readlines():
            line = line.rstrip().lower()
            
            splitted = line.split('|')
            splitted = [x.strip().rstrip().lower() for x in splitted]
            if len(splitted) == 2:
                if splitted[1] =='label':
                    continue
                label = int(splitted[0])
                name = splitted[1].rstrip().lower()
                if name  not in label_mappings.keys():
                    label_mappings[name] = []
                    label_mappings[name].append(label)
                else:
                    label_mappings[name].append(label)
            else:
                break
    return label_mappings
                




def generate_labels(header=['sensor_type','device','label','label_start','datetime_start','label_end','datetime_end'],
                    fname='./ml4qsbeasts/log_events',log_fname='log1',fout='./ml4qsbeasts/labels.csv'):
    
    label_mapping = get_labels(log_fname)
    labels_list = []
    lines_list = []
    with open(fname,'r+') as f:
        for line in f.readlines():
            lines_list.append(line)
    
    for line,line2 in pairwise(lines_list):
 
        splitted = line.split('|')
        splitted = [x.rstrip().lower() for x in splitted]
        label,_,_,time_start = splitted

        splitted = line2.split('|')
        splitted = [x.rstrip().lower() for x in splitted]
        label,_,_,time_end = splitted
        for k in label_mapping.keys():
            if int(label) in label_mapping[k]:
                key = k
        
        labels_list.append(['interval_label','tommie smartphone',
                              key,time_start,pd.to_datetime(time_start,unit='ms'),
                            time_end,pd.to_datetime(time_end,unit='ms')])
        
    
    with open(fout,'w+') as f:
        f.write(','.join([str(x) for x in header])+'\n')
        for line in labels_list:
            line = ','.join([str(x) for x in line])
            f.write(line+'\n')
    return labels_list

            
def get_sensor_values(fname,device='log1'):
    '''Get sensor values function. Parses the .txt log and puts all sensors in a csv
        args:
            fname the filename of the sensor log
        
        output:
            sensor_dict a dictionary where the keys are the sensors and the values are label-vector-timestamp
    '''
    sensor_dict = {}
    device = fname.split('/')[len(fname.split('/'))-1]
    labels_dict = {}
    label_names = get_labels(fname)
    with open(fname,'r+') as f:
        for line in f.readlines():
            splitted = line.split('|')
            if len(splitted) != 4:
                #in case we are before our data to avoid parsing errors
                continue
            splitted = [x.rstrip().lower() for x in splitted]
            label_id,sensorname,vector,timestamp = splitted
            if sensorname == 'sensorname':
                continue
            if sensorname not in sensor_dict.keys():
                sensor_dict[sensorname] = []
            else:
                tmp_list = sensor_dict.get(sensorname)
                tmp_list.append([label_id,vector,timestamp])
                sensor_dict[sensorname] = tmp_list
            
            
    return sensor_dict

def generate_csv(values,header=['x','y','z','timestamp'],fname='./data/toy_data/csv/gyroscope.csv'):
    '''Creates a .csv at a specified path for a certain input'''
    '''
        args:
            values    the sensor data values , a list of lists in our case
            header    the header of the .csv . Default is label,x,y,z,timestamp 
            fname     the path and filename to save the .csv file
        
            
    '''
    with open(fname,'w') as f:
        f.write(''.join([x+',' for x in header])+'\n')
        for v in values:
            
#             if str(v[0]) != str(label_only):
#                 continue
            v[2] = float(v[2])
            x = str(v[1]).replace('[','').replace(']','').replace(' ','')
            line = '%s,%d\n'%(x,v[2])
            f.write(line)
              
   
