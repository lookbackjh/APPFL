import os
import torch
import torchvision
from sklearn.preprocessing import LabelEncoder
from ml100k_preprocess import Ml100k
import numpy as np
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from appfl.misc.data import (
    Dataset,
    iid_partition,
    class_noniid_partition,
    dirichlet_noniid_partition,
    iid_partition_recsys
)

class FM_Dataloader(data_utils.DataLoader):

    def __init__(self,x,y):
        self.x=x
        self.y=y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]



def label_encode(df):
    #df=self.df
    label_encoders=[]
    for column in df.columns:
        if column=="rating":
            continue
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders.append(le)
    return df,label_encoders


def get_ml100k(num_clients: int, client_id: int, partition_strategy: str = "iid", **kwargs):
    """
    Return the ML-100K dataset for a given client.
    :param num_clients: total number of clients
    :param client_id: the client id
    """
    # Get the download directory for dataset
    dir = os.getcwd() + "/datasets/RawData"
    print(dir)

    #Root download the data if not already available.
    ml100k_data = Ml100k(dir)
    data= ml100k_data.get_data()
    label_encoded_data,label_encoders=label_encode(data)
    
    
    train_data_y=label_encoded_data['rating']
    train_data_x=label_encoded_data.drop(columns='rating')
    train_x, test_x,train_y,test_y = train_test_split(train_data_x,train_data_y, test_size=0.3, random_state=10)
    train_x_np=np.array(train_x,dtype=np.int64)
    test_x_np=np.array(test_x)
    train_y_np=np.array(train_y,dtype=np.int64)
    test_y_np=np.array(test_y)
    train_y=train_y_np.reshape(-1,1)
    test_y=test_y_np.reshape(-1,1)

    train_dataset = Dataset(torch.tensor(train_x_np).long(),torch.tensor(train_y).float())
    test_dataset = Dataset(torch.tensor(test_x_np).long(),torch.tensor(test_y).float())
    #field_dims=np.max(train_data_x, axis=0)+1
    #kwargs.client_configs_model_configs.model_kwargs.field_dims=field_dims
    #print("field_dims:",field_dims)
    #kwargs.model_configs.model_kwargs.embed_dim=16


    # Partition the dataset

    train_datasets = iid_partition_recsys(train_dataset, num_clients)
    # I want to change train_datasets into longtensor
    #print("len train_datasets:",len(train_datasets))
    # elif partition_strategy == "class_noniid":
    #     train_datasets = class_noniid_partition(train_data_raw, num_clients, **kwargs)
    # elif partition_strategy == "dirichlet_noniid":
    #     train_datasets = dirichlet_noniid_partition(
    #         train_data_raw, num_clients, **kwargs
    #     )
    # else:
    #     raise ValueError(f"Invalid partition strategy: {partition_strategy}")

    return train_datasets[client_id], test_dataset