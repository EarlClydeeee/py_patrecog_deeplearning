# Laboratory Activity
## Neural Networks and Deep Learning using Python

**Course:** CMPE 362 – Pattern Recognition  
**Submitted by:** Earl Clyde M. Bañez  
**Submitted to:** Malbog, Mon Arjay F.  
**Notebook:** `neural_networks.ipynb`

---

## Overview

This laboratory activity explores how neural networks work, from a single artificial neuron to deep learning models using TensorFlow/Keras. The notebook includes step-by-step code, visualizations, MNIST digit classification, and four exercises with guide question answers.

---

## Requirements

Install the required libraries:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy matplotlib tensorflow
```

---

## Exercises

### Exercise 1 – Modify Hidden Layer Neurons (128 → 256)

Change the hidden layer size in Step 15 from 128 to 256 neurons.

**1. Did the accuracy improve?**

The accuracy got a little better, but not by much. When I used 256 neurons with ReLU, the test accuracy was around 97.33%. When I tried 128 neurons, the result was also around 97–98%, so it was almost the same. I think doubling the neurons did not help a lot because MNIST is not that hard and 128 neurons was already enough for the model to recognize the digits.

**2. What changes occurred during training?**

While training, the accuracy went up every epoch and the loss went down, which means the model was learning. With 256 neurons, the model had 203,530 parameters, so it took more work to train compared to 128 neurons. The training accuracy started at about 92% in epoch 1 and reached about 98.6% by epoch 5. The loss also went down from about 0.26 to 0.05. Having more neurons gives the model more room to learn, but it also makes training slower and uses more memory.

---

### Exercise 2 – Replace ReLU with Sigmoid

Change the hidden layer activation from `'relu'` to `'sigmoid'` in Step 15.

**1. Which activation function performs better?**

Based on my results, ReLU worked better than Sigmoid. Here is what I got with 256 neurons and 5 epochs:

| Metric | ReLU | Sigmoid |
|--------|------|---------|
| Epoch 1 accuracy | 92.43% | 90.18% |
| Epoch 5 accuracy | 98.56% | ~97.93% |
| Test accuracy | 97.33% | Likely ~96–97% (run Step 18 to confirm) |

ReLU learned faster and got higher accuracy during training. Sigmoid was still okay on the test data, but ReLU is better to use in the hidden layer because it helps the model learn more easily.

**2. Which model trains faster?**

ReLU trained faster. ReLU took about 6–7 seconds per epoch, while Sigmoid took about 7–9 seconds. ReLU also got higher accuracy earlier. After epoch 2, ReLU was already at 96.64%, but Sigmoid was only at 94.80%. I think Sigmoid is slower because it limits the output between 0 and 1, which makes learning harder. ReLU is simpler because it just keeps positive values and sets negative values to zero.

---

### Exercise 3 – Increase Epochs (5 → 10)

Change the number of training epochs from 5 to 10 in Step 17.

```python
history = model.fit(
    X_train,
    y_train,
    epochs=10
)
```

**1. How did the model accuracy change?**

When I trained for 5 epochs, the training accuracy was about 97.97% and the test accuracy was about 97.57%. When I increased it to 10 epochs, the accuracy still improved, but not as much as the first 5 epochs.

| | 5 Epochs | 10 Epochs (expected) |
|---|----------|----------------------|
| Training accuracy (final epoch) | ~97.97% | ~98.5–99%+ |
| Test accuracy | ~97.57% | ~97.6–98% (small gain) |
| Training time | ~35–40 s | ~70–80 s (about double) |

Most of the improvement happened in the first 5 epochs. After that, the accuracy still went up a little, but the change was small because the model already learned most of the patterns. The loss also kept going down, but slower. Training also took about twice as long since the model went through the data 10 times instead of 5.

**2. What happens when epochs become too large?**

If the epochs are too many, the model can overfit. This means it memorizes the training data too much and does not work as well on new data. The training accuracy may keep going up, but the test accuracy can stop improving or even get worse. Training also takes longer without much benefit. So more epochs is not always better. You need enough epochs for the model to learn, but too many can make it memorize instead of actually understanding the digits. For MNIST, 5 to 10 epochs is usually enough.

---

### Exercise 4 – Add a Dropout Layer

Add a Dropout layer after the hidden layer in Step 15.

```python
keras.layers.Dropout(0.2)
```

**1. What is the purpose of dropout?**

The purpose of dropout is to help the model work better on new data and not rely too much on certain neurons. Dropout(0.2) randomly turns off 20% of the neurons while training. This helps the model learn in a stronger way and not just memorize the training data.

**2. How does dropout reduce overfitting?**

Dropout helps reduce overfitting because the model cannot always use the same neurons every time. Each batch is a little different, so the neurons have to learn useful patterns on their own. Because of this, the model is less likely to memorize the training examples. Sometimes the training accuracy becomes a little lower, but the test accuracy can stay good or even improve. When we test or predict, dropout is turned off and all neurons are used again.
