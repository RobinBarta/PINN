'''
    Enter the parameter and run the script.
    The data structure of the dataset is build. 
'''

import os 

# %%

class Dataset_parameter:
    NameOfDataSet = 'YOUR_CASENAME' # enter your casename here

# %%

def main():
    params = Dataset_parameter()
    
    # if the path does not exist, create dataset structure
    path = '../../../data/' + params.NameOfDataSet
    if not os.path.exists(path):
        os.mkdir( path )
        os.mkdir( path + "/input" )
        os.mkdir( path + "/input/raw_data" )
        os.mkdir( path + "/output" )
if __name__ == "__main__":
    main()
