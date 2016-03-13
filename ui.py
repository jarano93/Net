import pickle

def load_net(fName):
    bin_file = file(fName, 'rb')
    return pickle.load(bin_file)

def save_output(fName, string):
    out = open(fName, 'a')
    out.write(string)
    out.close()
