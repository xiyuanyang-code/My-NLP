from src.trainer import Word2VecTrainer

def main():
    trainer = Word2VecTrainer(data_path="hm1/data/ptb.train.txt")
    embed_sizes = [20, 100, 200, 500, 1000]
    lrs = [0.001, 0.0001, 0.005]
    num_epochs = 50
    freq = 10
    for embed_size in embed_sizes:
        for lr in lrs:
            trainer.train(embed_size=embed_size, lr=lr, num_epochs=num_epochs, save_freq=freq)

if __name__ == "__main__":
    main()