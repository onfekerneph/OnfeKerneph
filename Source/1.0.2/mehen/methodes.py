from core.network import Network
from train_datas import vocab_size,data,encode,id2word
import os
path = os.path.dirname(os.path.abspath(__file__))
def createModel(vocab_size,data,epochCount=2000,s=100):
    os.system("clear")
    print("=== Create Model ===")
    net = Network(
        vocab_size=vocab_size,
        dim=128,
        lr=0.01,
        ff_hidden=128,
    )
    print("=== Training ===")
    loss = net.train(data,
    epochs=epochCount+1,
    step=s,
    )

    net.save(path+"model.qai")
    return net
    
def startModel(net):
    import os
    os.system("clear")
    print("=== Run ===")
    while True:
        text = input("-> ").strip().lower()
        if text == "exit":
            break
        seq = encode(text)
        if not seq:
            print("Unknown input")
            continue
        generated = text
        for _ in range(200):
            pred = net.predict(seq)
            word = id2word[pred]
            if word == "esc":
                break
            print("Bot:", word)
            generated += " " + word
            seq = encode(generated)

