from methodes import *
from train_datas import data
if __name__ == "__main__":
    net = createModel(vocab_size,s=5,data=data)
    net.save(path+"/model.qai")