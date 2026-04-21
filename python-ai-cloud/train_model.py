from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import os
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import 
from sklearn.model_selection import train_test_split

def train_and_save_model()-> None:
    data = load_iris()
    X = data.data
    y = data.target
    #X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.2,random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X,y)
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)

    with open(model_dir / "model.pkl","wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved to model/model.pkl")

if __name__ == "__main__":
    train_and_save_model()