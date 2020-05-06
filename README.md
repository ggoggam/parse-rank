# parse-rank

## Parse reranking language model using LSTM. 

The model learns on the tree representation of sentence parses to predict the parses of unseen sentences by sequentially predicting the tokens.

<img src="https://render.githubusercontent.com/render/math?math=\mathbb{P}(\mathbf{x}) = \prod_{t=1}^T \mathbb{P}(x_t \mid x_1, \dots, x_{t-1})">

Note that the model trains and validates on correct parsing dataset, then tests on the different parses of the same sentence, reranking them according to correctness.

`reranker_train.txt` includes train and test set, `conv.txt` includes test parses, and `gold.txt` includes correct (gold) parses for sentences in `conv.txt`.
The metrics used for testing are recall, precision, and F1 score.

```
# Run training -> validation -> save
$ python parse_reranker.py -Tvs ./data/reranker_train.txt ./data/conv.txt ./data/gold.txt
```

More details on the sentence parsing and the architecture can be found in this [paper](http://cs.brown.edu/~dc65/papers/emnlp16.pdf). 
