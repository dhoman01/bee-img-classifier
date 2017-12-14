## Bee Image Classification

### Training
Place you data into your desired training dir (default is data/train/[single_bee_train|no_bee_train]). Place your test data into your desired testing dir (default is data/train/[single_bee_test|no_bee_test]). Finally, run `python3 train.py` to train the model. Inspect `train.py` to see availble flags.

### Classification
Place the model checkpoint into you desired dir (default is data/ckpt/). Run `python3 classifier.py --input_file /path/to/your/input_file.png`

### Accuracy
The provided checkpoint gives an accuracy of 97.7%
