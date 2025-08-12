import torch
import numpy as np
import pandas as pd
from DenseWideNet import DWN_feature
from collections import OrderedDict
import torch.nn as nn 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_single(wd, bk, lnum=1, fcNum=32):
        model = nn.Sequential(OrderedDict([
            ('fe', DWN_feature(wd, bk, lnum)),
            ('mv', nn.Linear(fcNum, 1)),
        ]))
        return model

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
        print('input_tensor.shape:', input_tensor.shape)
        with torch.no_grad():
            predictions = self.model(input_tensor)
        return predictions.cpu().numpy()

def input_reorganize(x):
    d=x.shape[1]
    x1 = x[:,0:15]
    x2 = x[:,15:d]
    x2_mean=np.reshape(x2,[x2.shape[0],int(d/46),46])
    x1=np.concatenate([x1,np.nanmean(x2_mean,axis=2),x2],axis=1)
    #x1=np.concatenate([x1,x2_mean[:,:,45],x2],axis=1)
    #x2= np.transpose(x2,(0,2,1))
    return x1

if __name__ == "__main__":
    blocks = 2
    width = 8
    # Example usage
    model_path = f"Demo/ft12_models/fusion_2020/a70_bauto_r0/m_{blocks}_{width}.pt"
    input_csv = "100m_data/output_tb/tb_merged.csv"
    output_csv = "data_pre/output_result.csv"

    # Load input data
    input_data = pd.read_csv(input_csv)
    input_data = np.asarray(input_data.loc[:, '0':'152']).astype(np.float32)
    input_data = input_reorganize(input_data)
    print('input_data.shape:', input_data.shape)

    # Initialize inference class
    inference = DenseSMInference(model_path, width=width, blocks=blocks)

    # Run inference
    predictions = inference.run_inference(input_data)

    # Save predictions with locations
    results_df = pd.DataFrame({
        # "location": locations,
        "Prediction": predictions.flatten()
    })
    results_df.to_csv(output_csv, index=False)

    print(f"Inference completed. Results saved to {output_csv}.")
