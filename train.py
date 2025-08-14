import pandas as pd
import utils
import os
import ft_ensemble
import numpy as np
import ee

# Path to the pretrained models folder
path_2_9km_models = 'pretrained_models/DenseSM_9km'

# path to training data CSV file
fusioned_data_path = '/mnt/data2tb/Transfer-DenseSM-E_pack/training_data/fusion/fusion_balanced.csv'
input_fine = pd.read_csv(fusioned_data_path, index_col='s_index')
# Drop all rows where 'sm' (soil moisture) values are greater than 0.7 
input_fine = input_fine[input_fine['sm'] <= 0.7]
input_fine = input_fine[input_fine['sm'] > 0.0]

# Print names of sub dataset used for training
np.unique(input_fine['network'])

# Define the setup parameters for training
# These parameters can be adjusted based on the specific requirements of the training process.
setup = {'lr':4e-4, # learning rate
        'epoch_Num':500,# number of epochs
        'swa_start':40,
        'alpha':0.7,# alpha in eq. 3
        'beta':'auto',# beta in eq. 3 and was determined by eq.4
        'domain_type':'coral',
        'mv_type':'MAPE',
        'ex':'ft12_models',
        'batchS':1024,# size of batch
        'br':1}# control the numbmer of unlabled samples and 9km samples, 1 means 1*batch_size
network_name=['CHINA_100m', 'CHINA_1km', 'INDIA_100m', 'INDIA_1km', 'VN']

# Folder storing trained models
base_dir = 'trained_models' #%network_name
model_dir_e = os.path.join(base_dir,setup['ex'])
os.makedirs(model_dir_e, exist_ok=True)

# Print the beta value from the setup dictionary
if isinstance(setup['beta'], str):
    print(setup['beta'])

from sklearn.model_selection import train_test_split

# name folder to save the models
nsample = 'fusion_balanced'
model_dir = os.path.join(model_dir_e, '%s'%nsample)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# 3 is the number of repeats (we want 3 different set of models)
for r in range(3):
    if isinstance(setup['beta'], str):
        model_dir_r = os.path.join(model_dir, 'a%s_b%s_r%s'%(int(setup['alpha']*100), setup['beta'], str(r)))
    else:
        model_dir_r = os.path.join(model_dir,'a%s_b%s_r%s'%(int(setup['alpha']*100),int(setup['beta']*100),str(r)))
    if not os.path.exists(model_dir_r):
        os.mkdir(model_dir_r)
    s_index = input_fine.index.to_list()

    # Split the dataset into train and val set (9:1)
    train_index, test_index = train_test_split(s_index, test_size=0.1, random_state=10, shuffle=True)

    # Print the number of samples in train and test sets
    print(len(train_index))
    print(len(test_index))

    # Prepare the training and validation data
    train_val_index = [train_index, test_index]
    trainloader, val_data, targetloader, train_data = utils.prepare_train_val_data_highres(input_fine, train_val_index, setup['batchS'], setup['br'])
    
    # Build the DenseSM models from the pretrained models
    modelX = ft_ensemble.Build_DenseSM(path_2_9km_models)
    modelX.rebuild_DenseSME()
    data = {'sl': trainloader,
            'tl':targetloader,
            'cl':None,
            'val_data': val_data,
            'train_data':train_data}
    
    # Prepare the setup for finetuning
    print(f'Finetune sample: {nsample} repeat: {r}')
    ft = ft_ensemble.FinetuneModel(setup, data)
    
    # Finetuning the model
    ft.ft_ensemble(modelX, model_dir_r)
    df, res, m_specific_y=ft_ensemble.ensemble_results(model_dir_r, val_data)
    res.scatter_density_fig(os.path.join(model_dir_r,'ensemble.jpg'))
    print(res.stat_3d)

