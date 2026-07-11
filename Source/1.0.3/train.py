from methodes import *
if __name__ == "__main__":
    net = createModel(vocab_size,data=data,s=1)
    net.save(path+"/model.qai")