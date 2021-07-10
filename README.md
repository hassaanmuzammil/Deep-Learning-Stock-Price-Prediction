# Deep-Learning-Stock-Price-Prediction

Open terminal and use the following command to clone the repository
```
git clone https://github.com/hassaanmuzammil/Deep-Learning-Stock-Price-Prediction.git
```


cd to the Deep-Learning-Stock-Price-Prediction
```
cd Deep-Learning-Stock-Price-Prediction
```

Create a virtual python 3.7.4 environment and make sure all requirements are satisfied
```
pip3 install -r requirements.txt
```

Check updated model.pth and dataset 2010 to 2021.csv files

Run the following python3 script
```
python3 inference.py --path="dataset/dataset 2010 to 2021.csv" --model="model/model.pth" 
```

The train and test predictions graph wrt actual stock prices is shown below

![image](https://user-images.githubusercontent.com/52124348/125101848-19894280-e0f4-11eb-8ef7-7b4f4c90dadc.png)


Results will be stored in results/results.csv which looks like this

![image](https://user-images.githubusercontent.com/52124348/125099694-bac2c980-e0f1-11eb-8d57-cee70c4faa9b.png)

The terminal output looks like this

![image](https://user-images.githubusercontent.com/52124348/125098732-ba75fe80-e0f0-11eb-85c7-d143ac9de615.png)

Refer to summary.txt for further details


