import pickle


with open(file='data.txt', mode='rb') as f:
    in_wb= pickle.loads(f.read())
