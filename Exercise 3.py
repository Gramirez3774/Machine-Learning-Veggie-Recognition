import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Creating dataset
num_samples = 1000
num_ft= 2

adults = [10,5]*np.random.randn(num_samples,num_ft) + [150,50]
target_adults = np.zeros(num_samples)
adults = np.vstack((adults.T,target_adults)).T

kids = [10,5]*np.random.randn(num_samples,num_ft) + [50,25]
target_kids = np.ones(num_samples)
kids = np.vstack((kids.T,target_kids)).T

dataset = np.concatenate((adults,kids))
df = pd.DataFrame(dataset, columns = ['Height','Weight','Target'])

print(dataset.shape)