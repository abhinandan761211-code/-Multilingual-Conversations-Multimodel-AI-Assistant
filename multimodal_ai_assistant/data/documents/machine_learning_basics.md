# Machine Learning Fundamentals

## Overfitting and Underfitting

Overfitting occurs when a model learns the training data too well, including its
noise and random fluctuations, so it fails to generalise to unseen data. The
classic symptom is a training loss that keeps decreasing while the validation
loss stops improving and then rises. The gap between training and validation
performance is the practical indicator of overfitting.

Underfitting is the opposite: the model is too simple to capture the underlying
pattern, so both training and validation loss remain high. A model that
underfits performs poorly even on the data it was trained on.

The standard remedies for overfitting are: collecting more training data,
reducing model capacity, adding regularisation (L1, L2, dropout), and early
stopping — halting training at the point where validation loss begins to rise.

## Training Loss vs Validation Loss

Training loss measures error on the data the model learns from. Validation loss
measures error on a held-out set the model never trains on. Early in training
both fall together. When validation loss starts to increase while training loss
continues to fall, the model has begun to memorise rather than learn. The epoch
at which validation loss reaches its minimum is typically the best point to stop
training (early stopping).

## Neural Networks

A neural network is a machine learning model inspired by interconnected
biological neurons. It is organised into layers:

- **Input layer** — receives the raw features. Its size equals the number of
  input features.
- **Hidden layers** — the intermediate layers between input and output. Each
  hidden layer applies a linear transformation followed by a non-linear
  activation function (ReLU, sigmoid, tanh). Hidden layers learn increasingly
  abstract representations: early layers capture simple patterns, deeper layers
  combine them into complex features.
- **Output layer** — produces the final prediction. For binary classification it
  is usually a single sigmoid unit; for multi-class it is a softmax over class
  scores.

The middle (hidden) layer's job is to transform the input representation into
something the output layer can separate. In a network diagram, the middle layer
is the one whose nodes do not connect directly to the input or output features.

## Activation Functions

An activation function introduces non-linearity, without which a deep network
would collapse into a single linear transformation. ReLU (rectified linear unit)
returns max(0, x) and is the default for hidden layers because it trains fast.
Sigmoid squashes values into (0, 1) and is used for probabilities. Softmax turns
a vector of scores into a probability distribution over classes.

## Gradient Descent and Learning Rate

Gradient descent updates parameters in the direction that reduces the loss. The
learning rate controls step size. A learning rate that is too high causes the
loss to oscillate or diverge; too low and training crawls or stalls in a plateau.

## Accuracy, Precision and Recall

Accuracy is the fraction of predictions that are correct. It is misleading on
imbalanced data. Precision answers "of the items we flagged positive, how many
truly were?" Recall answers "of the truly positive items, how many did we find?"
The F1 score is the harmonic mean of precision and recall.
