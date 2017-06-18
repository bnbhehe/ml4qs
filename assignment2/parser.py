import os
import json
import ast

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
                


def get_sensor_values2(fname,device='log1'):
    '''Get sensor values function. Parses the .txt log and puts all sensors in a csv
        args:
            fname the filename of the sensor log
        
        output:
            sensor_dict a dictionary where the keys are the sensors and the values are label-vector-timestamp
    '''
    sensor_dict = {}
    device = fname.split('/')[len(fname.split('/'))-1]
    print('The device is %s'%device)
    labels_dict = {}
    label_names = get_labels(fname)
    print(label_names)
    with open(fname,'r+') as f:
        for line in f.readlines():
            splitted = line.split('|')
            if len(splitted) != 4:
                #in case we are before our data to avoid parsing errors
                continue
            splitted = [x.rstrip().lower() for x in splitted]
            label_id,sensorname,vector,timestamp = splitted
            if label not in labels_dict.keys():
                labels_dict[label] = []
            else:
                if label == 'statusid':
                    continue
                tmp_list = labels_dict.get(label)
                tmp_list.append(['interval_label',
                                 device,
                                 label_names.keys()[label_names.values().index(int(label))],
                                 timestamp
                                ])
            ### same as above this is actually the header  list.
            if sensorname == 'sensorname':
                continue
            vector = ast.literal_eval(vector)
            if sensorname not in sensor_dict.keys():
                sensor_dict[sensorname] = []
            else:
                tmp_list = sensor_dict.get(sensorname)
                tmp_list.append([label,vector,timestamp])
                sensor_dict[sensorname] = tmp_list
            
    return sensor_dict,labels_dict

def generate_labels2(values,header=['device','label','label_start','label_end'],fname='./data/toy_data/csv/labels.csv'):
    label_mapping = get_sensor_values(fname)
    
    with open(fname,'r+') as f:
        for line in f.readlines():
            splitted = line.split('|')
            if len(splitted) != 4:
                continue
            splitted = [x.rstrip().lower() for x in splitted]
            
              
    
def generate_csv2(values,header=['x','y','z','timestamp'],label_only='2',fname='./data/toy_data/csv/gyroscope.csv'):
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
            if header[0] =='light_intensity':
                x = x.replace(',0.0','')    
            line = '%s,%d\n'%(x,v[2])
            f.write(line)

            
def generate_labels(values,header['device','label','label_start','label_end'],
                    fname='./ml4qsbeasts/log_events',fout='./ml4qsbeasts/labels.csv):
    label_mapping = get_sensor_values(fname)
    with open(fname,'r+') as f:
        with open(fout,'w+') as f2:
            
                  
              

            
def get_sensor_values(fname,device='log1'):
    '''Get sensor values function. Parses the .txt log and puts all sensors in a csv
        args:
            fname the filename of the sensor log
        
        output:
            sensor_dict a dictionary where the keys are the sensors and the values are label-vector-timestamp
    '''
    sensor_dict = {}
    device = fname.split('/')[len(fname.split('/'))-1]
    print('The device is %s'%device)
    labels_dict = {}
    label_names = get_labels(fname)
    print(label_names)
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