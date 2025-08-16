import torch
import numpy as np
import pandas as pd
from DenseWideNet import DWN_feature
from collections import OrderedDict
import torch.nn as nn 
import os
import argparse 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--region', required=True)
args = parser.parse_args()

# Build the model from the given width and number of blocks
def build_single(wd, bk, lnum=1, fcNum=32):
        model = nn.Sequential(OrderedDict([
            ('fe', DWN_feature(wd, bk, lnum)),
            ('mv', nn.Linear(fcNum, 1)),
        ]))
        return model

# Rebuild the DenseSME model with the given width and blocks, and load the model state
def rebuild_DenseSME(wd, bk, temp_model_state):
        model = build_single(wd, bk)
        try:
            model.fe.load_state_dict(temp_model_state, strict=False)
            model.mv.weight.data = temp_model_state['out.weight']
            model.mv.bias.data = temp_model_state['out.bias']
        except:
            model.load_state_dict(temp_model_state)

        return model 

class DenseSMInference:
    def __init__(self, model_path, width, blocks, fc_num=32):
        # self.model = self._load_model(model_path, width, blocks, fc_num)
        self.model_state = torch.load(model_path, map_location=device)
        self.model = rebuild_DenseSME(width, blocks, self.model_state)
        self.model.to(device)
        self.model.eval()

    def _load_model(self, model_path, width, blocks, fc_num):
        model = torch.nn.Sequential(OrderedDict([
            ('fe', DWN_feature(width, blocks, fc_num = 1)),
            ('mv', torch.nn.Linear(fc_num, 1)),
        ]))
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    def run_inference(self, input_data):
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
        # print('input_tensor.shape:', input_tensor.shape)
        with torch.no_grad():
            predictions = self.model(input_tensor)
        return predictions.cpu().numpy()

def input_reorganize(x):
    d=x.shape[1]
    x1 = x[:,0:15]
    x2 = x[:,15:d]
    x2_mean=np.reshape(x2,[x2.shape[0],int(d/46),46])
    x1=np.concatenate([x1,np.nanmean(x2_mean,axis=2),x2],axis=1) # Get the mean values of NDVI, Temperature, Precipitation , better
    #x1=np.concatenate([x1,x2_mean[:,:,45],x2],axis=1) #Get the last values of NDVI, Temperature, Precipitation
    #x2= np.transpose(x2,(0,2,1))
    return x1


region = args.region
# Define the model variants to be used for inference, distinguish by (number of blocks, width)
models = [(1,8),(1,16),(1,32),(1,64),(1,128),(2,8),(2,16),(2,32),(2,64),(2,128),(3,8),(3,16),(3,32),(3,64),
(3,128),(4,8),(4,16),(4,32),(4,64),(4,128),(5,8),(5,16),(5,32),(5,64),(5,128)]

# Path to the folder containing the CSV files output of merge_and_process.py for each date
combination_folder = f'roi_inference/regions_data_results/{region}/csv_output/combination'

# Get all dates need to be predicted 
target_dates = []
for s1_file in os.listdir(combination_folder):
    date = s1_file.split('.')[0].split('_')[0]
    target_dates.append(date)

# Traverse through the file of each date and run inference
for target_date in target_dates:
    input_csv = f"roi_inference/regions_data_results/{region}/csv_output/combination/{target_date}_tif.csv"
    print("Input file ", input_csv)
    output_csv = f"roi_inference/regions_data_results/{region}/prediction/{target_date}.csv"
    if os.path.exists(output_csv):
            print(f'{target_date} already predicted')
        #  continue

    # Load input data
    input_data = pd.read_csv(input_csv)
    # Check if columns '0' to '152' exist
    expected_cols = [str(i) for i in range(0, 153)]
    missing_cols = [col for col in expected_cols if col not in input_data.columns]
    if missing_cols:
        print(f"Error: Missing columns in input_data: {missing_cols}")
        continue
    input_data = np.asarray(input_data.loc[:, '0':'152']).astype(np.float32)
    input_data = input_reorganize(input_data)
    records = []
    # Traverse through the model variants and run inference
    # For each model, load the input data, run inference and save the predictions
    # We apply ensemble by averaging the predictions of all models
    for bk, wd in models:
        model_path = f"trained_models/ft12_models/fusion_full/a70_bauto_r10/m_{bk}_{wd}.pt"
        # Initialize inference class
        inference = DenseSMInference(model_path, width=wd, blocks=bk)

        # Run inference
        predictions = inference.run_inference(input_data)
        # Save predictions
        records.append(predictions)

    # Average the predictions from all models
    # predictions.shape: (num_samples, 1) for each model, we stack them
    all_predictions = np.stack(records, axis = 0)
    average_predictions = np.mean(all_predictions, axis = 0)
    # Save predictions to CSV
    results_df = pd.DataFrame({
        # "location": locations,
        "Prediction": average_predictions.flatten()
    })
    results_df.to_csv(output_csv, index=False)

    print(f"Inference completed. Results saved to {output_csv}.")
