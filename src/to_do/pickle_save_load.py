import pickle
import pandas as pd

def to_pickle(input_file):
    print("Enter desired name:")
    name = input()
    pickle_out = open(name + '.pickle','wb')
    pickle.dump(input_file, pickle_out)
    pickle_out.close()


def import_pickle_as_pd(file_path):
    pickle_in1 = open(file_path,'rb')
    test_df=pickle.load(pickle_in1)
    return pd.DataFrame(test_df).T
