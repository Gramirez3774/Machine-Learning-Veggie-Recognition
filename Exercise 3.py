import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    print("Program Started")
    #Creating dataset parameters
    num_samples = 1000
    num_ft= 2
    #Create Adult dataset and adult targets
    adults = [10,5]*np.random.randn(num_samples,num_ft) + [150,50]
    target_adults = np.zeros(num_samples)

    #create kids dataset and kids targets 
    kids = [10,5]*np.random.randn(num_samples,num_ft) + [50,25]
    target_kids = np.ones(num_samples)

    #concatenate adults and kids dataset and targets. Appending rows from one matrix to the other.
    dataset = np.concatenate((adults,kids))
    target = np.concatenate((target_adults,target_kids))

    #Plot dataset and targets
    plt.scatter(dataset[:,0],dataset[:,1],c=target, s = 5)
    plt.title("Kids/Adult Classification")
    plt.xlabel("Height (cm)")
    plt.ylabel("Weight (kg)")
    plt.show()

    #Convert the numpy matrices to pandas dataframes
    df_dataset = pd.DataFrame(dataset, columns = ['Height','Weight'])
    df_targets = pd.DataFrame(target, columns = ['Target'])

    #export pandas dataframes to csv
    df_dataset.to_csv("dataset.csv")
    df_targets.to_csv("targets.csv")

    #imports csv to pandas dataframes
    df_imported_dataset= pd.read_csv("dataset.csv", index_col=0)
    df_imported_targets= pd.read_csv("targets.csv", index_col=0)

    #print dataframes
    print(df_imported_targets)
    print(df_imported_dataset)

    print("Program Ran Successfully")

if __name__ == "__main__":
    main()