from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

import pandas
import numpy
import torch

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
  
#Concatenating X and y to drop y rows with Nan values
X = pandas.DataFrame(numpy.concatenate((X, y), axis=1))

#Dropping the rows with Nan values
X = X.dropna()
y = pandas.DataFrame(X.iloc[:, -1])

#Making y binary
for i in (range(len(y))):
    if y.iloc[i, 0] != 0:
        y.iloc[i, 0] = 1

X = pandas.DataFrame(X.drop(13,axis='columns'))
#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
#Normalizing the data
sc = StandardScaler()
columns_to_scale = [0,3,4,7,9,11]

X_train[columns_to_scale] = sc.fit_transform((X_train[columns_to_scale]))
X_test[columns_to_scale] = sc.transform(X_test[columns_to_scale])

#Converting the data to torch tensor
X_train = torch.tensor(data=X_train.values, dtype=torch.float32)
X_test = torch.tensor(data=X_test.values, dtype=torch.float32)
y_train = torch.tensor(data=y_train.values, dtype=torch.float32)
y_Test = torch.tensor(data=y_test.values, dtype=torch.float32)

#Creating the Architecture of the NN
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        
        self.fc1 = torch.nn.Linear(in_features = X_train.shape[1], out_features = 32)
        self.fc2 = torch.nn.Linear(in_features=32, out_features=64)
        self.fc3 = torch.nn.Linear(in_features = 64, out_features = 1)
        
    #Applying forward-propagation
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.sigmoid(self.fc3(x))
        
        return x

#Creating the model
model = model()

#Adding the criterion and the optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
epochs = 200

#Training
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
#Testing the model
y_pred = model(X_test)   

y_pred = y_pred.detach().numpy()

#Making y_pred binary
for i in range(len(y_pred)):
    if y_pred[i] >= 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
    
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

f1 = f1_score(y_test, y_pred, average='macro')  # Macro-averaging
print("F1-score (Macro):", f1)
    
