import torch
import pandas as pd
from model import Net, Trainer
from data_manager import LoaderFactory, split_data_frame

MODEL_PATH = '../saved_model/model.pth'


TRAIN_DATA = '../data/train.csv'
TEST_DATA = '../data/test.csv'
TRAIN_IMG = '../data/train_test_data/train/'

if __name__ == '__main':
    # load data
    train_df, test_df = split_data_frame(pd.read_csv(TRAIN_DATA))
    loader_fact = LoaderFactory()
    train_DL = loader_fact.get_data_loader('../data/', train_df, train=True)
    test_DL = loader_fact.get_data_loader('../data/', train_df, train=True)

    # train splited data
    splited_net = Net()
    trainer = Trainer(splited_net, train_DL)
    trainer.train()
    y_pred, y_true = splited_net.predict(test_DL)
    score = splited_net.score(y_pred, y_true)
    print(f'Finished training, F1 score (macro) of {score}')
        # train
        # predict    
        # calcualte f1 score

    # train whole data
    # save model
        # torch.save(net.state_dict(), MODEL_PATH)    
    # predict+save predictions