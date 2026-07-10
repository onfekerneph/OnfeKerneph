from core.network import Network
from train_datas import vocab_size,data,encode,id2word

def createModel(vocab_size,epochCount=2000,s=100):
    net = Network(vocab_size=vocab_size,
    dim=64,lr=0.005,
    ff_hidden=128,)

    print("=== Training ===")

    loss = net.train(data,
    epochs=epochCount+1,
    step=s,
    )

    net.save("model.qai")
    return net
    
def startModel(net):
    print("===Run===")

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

if __name__ == "__main__":
    net = createModel(vocab_size)

    