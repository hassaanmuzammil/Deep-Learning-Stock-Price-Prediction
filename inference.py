#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go



def read_dataframe(dataset):
    """
    Reads a csv file to pandas dataframe with index as date values

    Args:
        file_path: string
            Representing path to file 

    Returns:
        Pandas dataframe
    """
    df = pd.read_csv(dataset)
    df = df.set_index('Date')
    
    return df



def preprocess_data(df, method):
    
    """
    Perform data normalization on the dataframe columns using methods from sklearn library 

    Args:
        df: dataframe
            Pandas DataFrame containing the stock prices info
        method: string
            Specifies which method to use 

    Returns:
        Processed dataframe
    """
    price = df[['Close']]
    
    if method in ['MinMax0']:
        scaler = MinMaxScaler(feature_range=(0, 1))
        price = scaler.fit_transform(price)
         
    elif method in ['MinMax1']:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        price = scaler.fit_transform(price)
        
    elif method in ['Standard']:
        scaler = StandardScaler()
        price = scaler.fit_transform(price)
    
    else:
        pass
    
    return price,scaler



def split_data(stock, lookback=20):
    """
    Lookback upto a certain number of days for training. Also splits the data into train test split.

    Args:
        stock: dataframe
            Pandas DataFrame containing the stock prices info
        lookback: int
            Specifies how many days to take into consideration for training 

    Returns:
        numpy arrays containing features and labels for both training and test
    """
    
    data_raw = stock # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data);
    test_set_size = 378
    train_set_size = 2517
    
    X_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    X_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [X_train, y_train, X_test, y_test]


def load_data(X_train, y_train, X_test, y_test):
    """
    Load data to tensor for model training/evaluation

    Args:
        X_train, y_train, X_test, y_test : numpy arrays

    Returns:
        corresponding tensor for all four arrays
    """
    
    X_train = torch.from_numpy(X_train).type(torch.Tensor)
    X_test = torch.from_numpy(X_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    
    return [X_train, y_train, X_test, y_test]




class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out


# In[7]:


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out




def initialize_load_model(model_type,model_file_path,load=True,input_dim=1,hidden_dim=256,output_dim=1,num_layers=1):
    """
    Initialize the model architecture

    Args:
        model_type: string
        Select 'GRU' or 'LSTM'
        
        input_dim, hidden_dim, num_layers, num_epochs: int
        Integers representing model features

    Returns:
        Pytorch model
        
    """
        
    if model_type == 'gru':
        model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        
    elif model_type == 'lstm':
        model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    if load == True:
        #load model state dict from file
        model.load_state_dict(torch.load(model_file_path))
    
    #defining loss and optimizer
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    return model,criterion,optimizer


def train_model(model,criterion,optimizer,X_train,y_train,num_epochs=100,show_graph=True):
    """
    Train the model to a given number of epochs

    Args:
        model: torch model
        
        X_train,y_train: torch tensors
        Train features and labels
        
        num_epochs: int
        
        show_graph: bool
        Whether to show results or not

    Returns:
        Trained Pytorch model
        
    """
    import time
    hist = np.zeros(num_epochs)
    start_time = time.time()

    for t in range(num_epochs):
        y_train_pred = model(X_train)
        loss = criterion(y_train_pred, y_train)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    training_time = time.time()-start_time
    print("Training time: {}".format(training_time))
    
    if show_graph:
        plt.figure(figsize=(20,5))

        plt.subplot(1,2,1)
        plt.plot(y_train_pred.tolist(),'r',label='Predicted')
        plt.plot(y_train.tolist(),'b',label='Actual')
        plt.title('Closing Stock Price')
        plt.legend()
        plt.xlabel('Training Period 01/01/2010 – 12/31/2019')
        plt.xticks([])
        plt.ylabel('Normalized Cost')

        plt.subplot(1,2,2)
        plt.plot(hist)
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.show()
    
    return model


def save_model(model,path):
    #save the trained model on local computer
    torch.save(model.state_dict(), path)
    
    pass



def get_predictions(model,X_train,y_train,X_test,y_test,scaler):
    #make predictions
    y_train_pred = model(X_train)
    y_test_pred = model(X_test)

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())
    
    return y_train_pred,y_test_pred,y_train,y_test


def test_model_performance(y_train_pred,y_test_pred,y_train,y_test,show_graph=False):
    """
    Perform metrics evaluation on the model

    """
    import math, time
    from sklearn.metrics import mean_squared_error

    # calculate root mean squared error
    print('\n\n------------------------------------------------------')
    print('The final train and test set root mean squared errors are:')
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    print('------------------------------------------------------\n\n')
    
    if show_graph:
        plt.figure(figsize=(10,5))

        plt.plot(y_test_pred.tolist(),'r',label='Predicted')
        plt.plot(y_test.tolist(),'b',label='Actual')
        plt.title('Closing Stock Price')
        plt.legend()
        plt.xticks([])
        plt.xlabel('Test Period 01/01/2020 – 02/07/2021')
        plt.ylabel('Normalized Cost')
        
    pass



def make_csv(df,y_train_pred,y_test_pred,filename,lookback=20):
    """
    Make csv file for the predicted, actual stock closing prices, and their differences
    
    Args:
        y_train_pred, y_test_pred
    Returns:
        dataframe from which csv file is constructed and saved on local computer
    """
    a = np.random.rand(20,1)
    y_pred = np.concatenate((y_train_pred,y_test_pred))
    y_pred = np.concatenate((a,y_pred))
    
    #print(len(y_pred))
    #print(len(df[['Close']]))
    
    df['Predictions'] = y_pred
    
    final_df = df[['Close','Predictions']]
    try:
        final_df['Difference'] = np.abs(np.array(final_df['Close']) - np.array(final_df['Predictions']))
    except:
        pass

    final_df.to_csv(filename)
    
    return final_df


def visualize_results(price,y_train_pred,y_test_pred,lookback,scaler):
    
    print("\n\n========================================================")
    print("Head over to the local host to see the graphical results")
    print("========================================================\n\n")
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(price)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(price)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(y_train_pred)+lookback-1:len(price)-1, :] = y_test_pred

    original = scaler.inverse_transform(price.reshape(-1,1))

    predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
    predictions = np.append(predictions, original, axis=1)
    result = pd.DataFrame(predictions)
    result = result.set_axis(['TrainPredictions','TestPredictions','ActualPrice'],axis=1,inplace=False)


    fig = go.Figure()
    fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result['TrainPredictions'],
                        mode='lines',
                        name='Train Predictions')))
    fig.add_trace(go.Scatter(x=result.index, y=result['TestPredictions'],
                        mode='lines',
                        name='Test Predictions'))
    fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result['ActualPrice'],
                        mode='lines',
                        name='Actual Price')))
    fig.update_layout(
        xaxis=dict(
            title_text='Time Period: 01/04/2010 - 02/07/2021',
            showline=True,
            showgrid=True,
            showticklabels=False,
            linecolor='white',
            linewidth=2
        ),
        yaxis=dict(
            title_text='Close (USD)',
            titlefont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='white',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Rockwell',
                size=12,
                color='white',
            ),
        ),
        showlegend=True,
        template = 'plotly_dark'

    )



    annotations = []
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                                  xanchor='left', yanchor='bottom',
                                  text='Stock Closing Price Prediction Results (GRU)',
                                  font=dict(family='Rockwell',
                                            size=26,
                                            color='white'),
                                  showarrow=False))
    fig.update_layout(annotations=annotations)
    
    fig.show()
    
    pass



def main(dataset,model_file_path):
    """
    Run the algorithm

    Args:
        model_file_path: trained torch model weights file
        
        file_path: dataset file path

    Returns:
        Visual results of stock prediction
        
    """
    
    #Read the csv file as pandas dataframe
    df = read_dataframe(dataset)
    
    #Normalize the entries of the dataframe based on a scaling method
    price,scaler = preprocess_data(df,method='Standard')
    
    #Train-test split the data
    X_train, y_train, X_test, y_test = split_data(price)
    
    #Load data as torch tensors
    X_train, y_train, X_test, y_test = load_data(X_train, y_train, X_test, y_test)
    
    #print(X_train.shape)
    #print(y_train.shape)
    #print(X_test.shape)
    #print(y_test.shape)
    
    #Initialize the model architecture and load state dict from path
    model,criterion,optimizer = initialize_load_model('gru',model_file_path)
    
    #print(model)
    
    #Train the model 
    #model = train_model(model,criterion,optimizer,X_train,y_train,num_epochs=100,show_graph=True)
    
    #Save the trained model state dict
    #save_model(model,path='local path')
    
    #Get train and test predictions from the trained model
    y_train_pred,y_test_pred,y_train,y_test = get_predictions(model,X_train,y_train,X_test,y_test,scaler)
    
    #Evaluate model performance 
    test_model_performance(y_train_pred,y_test_pred,y_train,y_test,show_graph=False) 
    
    #store the final results in csv on local computer
    final_df = make_csv(df,y_train_pred,y_test_pred,filename="results.csv")
    
    #Visualize the predictions
    visualize_results(price,y_train_pred,y_test_pred,lookback=20,scaler=scaler)
    
    return 0



if __name__== '__main__':
    
    print('\n===========================================================================================================')
    print('The program will attempt to predict Stock Prices from the dataset given for the duration 01/04/2010 - Present')
    print('===========================================================================================================\n')
    
    
    dataset = "dataset/dataset 2010 to 2021.csv"
    model_file_path = "model/model.pth"
    
    main(dataset,model_file_path)
    



