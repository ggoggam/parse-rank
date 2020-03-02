from comet_ml import Experiment
from preprocess import ParsingDataset, RerankingDataset
from model import LSTMLM
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar

# TODO: Set hyperparameters
hyperparams = {
    "rnn_size": 256,
    "embedding_size": 64,
    "num_epochs": 4,
    "batch_size": 50,
    "learning_rate": 1e-2
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



def train(model, train_loader, experiment, hyperparams):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    model = model.train()
    with experiment.train():
        for _ in range(hyperparams['num_epochs']):
            for batch in tqdm(train_loader):
                x = batch['inputs'].to(device)
                y = batch['labels'].to(device)
                l = batch['length'].to(device)
                
                optimizer.zero_grad()
                logits = model(x, l)
                y = y[:logits.shape[0]]
                losses = loss_fn(logits.flatten(0,1), y.flatten(0,1))
                losses.backward()
                optimizer.step()



def validate(model, validate_loader, experiment, hyperparams):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0
    word_count = 0

    model = model.eval()
    with experiment.validate():
        with torch.no_grad():
            for batch in tqdm(validate_loader):
                x = batch['inputs'].to(device)
                y = batch['labels'].to(device)
                l = batch['length'].to(device)

                logits = model(x, l)
                y = y[:logits.shape[0]]

                total_loss += loss_fn(logits.flatten(0,1), y.flatten(0,1)) * torch.sum(l)
                word_count += torch.sum(l)
        
        perplexity = torch.exp(torch.tensor(total_loss.item() / word_count.item()))        
        print("perplexity:", perplexity)
        experiment.log_metric("perplexity", perplexity)



def test(model, test_dataset, experiment, hyperparams):
    model = model.eval()
    with experiment.test():
        with torch.no_grad():
            sentence_prob, sentence_id = [], []
            correct_, total_, actual_ = [], [], []
            for batch in test_dataset:
                x = batch['inputs'].to(device)
                l = batch['length'].to(device)

                sentence_id.extend([x.item() for x in batch['id']])
                correct_.extend([x.item() for x in batch['num_correct']])
                total_.extend([x.item() for x in batch['num_total']])
                actual_.extend([x.item() for x in batch['num_actual']])

                out = model(x, l)
                for s, o in zip(x, out):
                    t = torch.squeeze(torch.gather(o, 1, s.unsqueeze(-1)))
                    sentence_prob.append(torch.sum(torch.log(t)).item())

            # Find maximal prob
            correct, total, actual = 0, 0, 0
            num_sentence = max(sentence_id)
            for i in range(num_sentence):
                idx = [x for x in range(len(sentence_id)) if sentence_id[x] == i]
                prob = [sentence_prob[x] for x in idx]

                max_prob_idx = prob.index(max(prob))

                correct += [correct_[x] for x in idx][max_prob_idx]
                total += [total_[x] for x in idx][max_prob_idx]
                actual += [actual_[x] for x in idx][max_prob_idx]

            precision = correct / total
            recall = correct / actual
            f1 = (2 * precision * recall) / (precision + recall)

            print("precision:", precision)
            print("recall:", recall)
            print("F1:", f1)
            experiment.log_metric("precision", precision)
            experiment.log_metric("recall", recall)
            experiment.log_metric("F1", f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("parse_file")
    parser.add_argument("gold_file")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loo      p")
    parser.add_argument("-v", "--validate", action="store_true",
                        help="run validation loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    args = parser.parse_args()

    # Make sure you modify the `.comet.config` file
    API_KEY = None
    PROJ_NAME = None
    WORKSPACE = None
    experiment = Experiment(api_key=API_KEY,
                            project_name=PROJ_NAME,
                            workspace=WORKSPACE)
    experiment.log_parameters(hyperparams)

    # Load Dataset
    pd = ParsingDataset(args.train_file)
    rd = RerankingDataset(args.parse_file, args.gold_file, pd.word2id)

    # Some variable declaration
    train_size = int(0.9 * len(pd))
    validate_size = len(pd) - train_size
    vocab_size = len(pd.word2id)
    max_length_train = torch.max(pd.dataset['length'])
    max_length_test = torch.max(rd.dataset['length'])

    # Create DataLoader
    train_, validate_ = random_split(pd, [train_size, validate_size])
    train_loader = DataLoader(dataset=train_,
                              batch_size=hyperparams['batch_size'],
                              shuffle=True)
    validate_loader = DataLoader(dataset=validate_,
                                 batch_size=hyperparams['batch_size'],
                                 shuffle=True)
    test_dataset = DataLoader(dataset=rd,
                              batch_size=hyperparams['batch_size'],
                              shuffle=False)

    # Create model
    model = LSTMLM(vocab_size,
                   hyperparams["rnn_size"],
                   hyperparams["embedding_size"]).to(device)

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        print("running training loop...")
        train(model, train_loader, experiment, hyperparams)
    if args.validate:
        print("running validation...")
        validate(model, validate_loader, experiment, hyperparams)
    if args.test:
        print("testing reranker...")
        test(model, test_dataset, experiment, hyperparams)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
    
    experiment.end()
