import inout
import pandas as pd
from model import Net, Trainer
from data_manager import LoaderFactory, split_data_frame


TRAIN_DATA = '../data/train.csv'
TEST_DATA = '../data/test.csv'
TRAIN_IMG = '../data/train_test_data/train/'

if __name__ == '__main__':
    # load data
    train_df, test_df = split_data_frame(pd.read_csv(TRAIN_DATA))
    loader_fact = LoaderFactory()
    train_DL = loader_fact.get_data_loader('../data/', train_df, batch_size=64)
    test_DL = loader_fact.get_data_loader('../data/', train_df, batch_size=64)


    # train splited data
    splited_net = Net((3, 332, 332), classes=3)
    trainer = Trainer(splited_net, train_DL, epochs=30)
    trainer.train()
    print('Predicting...')
    y_pred, y_true = splited_net.predict(test_DL)
    print('Calculating score...')
    score = splited_net.score(y_pred, y_true)
    print(f'F1 score (macro) of {score}')
        

    # train whole data
    full_df = pd.read_csv(TRAIN_DATA)
    final_test_df = pd.read_csv(TEST_DATA)
    full_DL = loader_fact.get_data_loader('../data/', full_df, batch_size=64)
    final_test_DL = loader_fact.get_data_loader('../data/', final_test_df, batch_size=64, train=False)
    final_net = Net((3, 332, 332), classes=3)
    trainer = Trainer(final_net, full_DL, epochs=30)
    trainer.train()
    predictions = final_net.predict_not_labeled(final_test_DL)
    inout.save_as_json('../predictions/predictions.json', predictions.detach().numpy())
    