# Author: Ashish Chokhani

# Title: Encoder Decoder model with Transformer and Attention Mechanisms
--- 

## Hey everyone! In this project I have implemented an Encoder Decoder model for NLP task.
- Training data and eval data have been provided in data directory.
- If you go through the data, you would find that input is sequence of characters of fixed length(8) and similarly, output is another sequence of characters of same length.
- Now initially to achieve the translation, I tried with different model such as RNN, GRU, LSTM and Bidirectional LSTM.
- It turned out that Bidirectional LSTM was the best choice as even though the length of input and transformed sentence is pretty small, but it had long term dependencies which Bidirectional LSTM could solve.
- However the implementation involved tensorflow. Also it did not run on GPU and hence it required 2 hrs to train the model :(.
- Again the accuracy rate with that implemnntation was around 94%.
- But later I also added Transformer and Attention Mechanism.
- It worked magically well :). Our model was able to achieve accuracy rate of 98%.
- Also its implementation was much faster as I had converted the code to Pytorch Implementation instead of Tensorflow and hence the entire code ran on GPU/CUDA.
- Feel free to reach me if you have any suggestions.

### GRUs

![alt text](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/10/Screen-Shot-2015-10-23-at-10.36.51-AM.png “WildML”)

- Use tanh instead of sigmoid

- Has two gates: an Update Gate, and a Reset Gate

- Update Gate:

$$z_{t} = {\sigma}(W^{(z)}x_{t} + U^{(z)}h_{t-1})$$

- Reset Gate:

$$r_{t} = {\sigma}(W^{(r)}x_{t} + U^{(r)}h_{t-1})$$

- New memory content, as a combination of new input and a fraction of old memory:

$$hh_{t} = tanh(Wx_{t} + r .* Uh_{t-1})$$

- Updated memory content, as a combination of fraction of old memory content and complementary new memory content:

$$h_{t} = z_{t} .* h_{t-1} + (1 - z_{t}) .* hh_{t}$$

- We can see that if z_{t} is close to 1, we can retain older information for longer, and avoid vanishing gradient.

### LSTMs

- LSTMs have 3 gates - Forget Gate, Input Gate, and Output Gate

![alt text](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png] “Colah’s”)

### Bi-directional RNNs

### Stacking RNNs

- Many flavours of Sequence-to-Sequence problems

- One-to-one (image classification), one-to-many (image captioning), many-to-one (video classification), asynchronous many-to-many (language translation), synchronous many-to-many (video labelling)

### RNN
- [input, previous hidden state] -> hidden state -> output

- RNNs model the joint probability over the sequences as a product of one-step-conditionals (via Chain Rule)

- Each RNN output models the one-step conditional $p(y_{t+1} | y_{1}, … , y_{t})$

### ENCODER-DECODER FRAMEWORK

- [Sutskever et al., 2014](); [Cho et al., 2014]()

- Can stack RNNs together, but in my experience any more than 2 is unnecessary

- Thang Luong’s Stanford CS224d lecture

- Loss function: Softmax + Cross-Entropy

- Objective is to maximize the joint probability, or minimize the negative log probability

- The encoder is usually initialized to zero

- If a long sequence is split across batches, the states are retained

### REPRESENTATION: Feature Embeddings / One-Hot Encoding

#### Domain-specific features
    - ConvNet fc feature vectors for images
    - Word2Vec features for words
    - MFCC features for audio / speech
    - PHOC for OCR (image -> text string)

#### One-hot encoding

### Word-level
    - Usually a large lexicon of ~100k words
    - Cumbersome to handle
    - Softmax is unstable to train with such huge fan out number

- So we go for:

### Character-level
    - Represent as sequence of characters

### INFERENCE

- We don’t take argmax of the output probabilities because we will not optimize the joint probability then.

- Exact inference is intractable, since exponential number of paths with sequence length

- Why can’t we use Viterbi Decoder as in HMMs?***

### Beam Search with Approximate Inference

- So, we compromise with an Approximate Inference:
    - We do a Beam Search through the top-k output classes per iteration (k is usually ~20)
    - So, we start with the <start> token -> take the top-k output classes -> use each of them as the next input -> get the output class scores for each of the k potential sub-sequences -> sum the scores and take the top-k output classes -> use each of them as the next input …

### LANGUAGE MODELLING

- Use RNN so as to capture context

### WHAT HAS RNN LEARNT?

- Interpretable Cells
    - Quote-detection cell: one value of the hidden state is 0 when the ongoing sentence is within quotes, 0 else
    - Line length tracking cell: gets warmer with length of line
    - If statement cell
    - Quote/comment cell
    - Code depth cell (indentation)

### ATTENTION MECHANISM

- Compare source and target hidden states
- Score the comparison between the hidden states of a source and a target node -> Do this for all encoder nodes with one target node (Make scores) -> Scale them and normalize w.r.t. Each other (Make Alignment Weights) -> Weighted Average

- [Bahdanau at al., 2015 (attention mechanism)](https://arxiv.org/abs/1409.0473)
    - Example of a well-written paper

### Text Image to Text String (OCR)
    - Recurrent Encoder-Decoder with Attention
    - Fully convolutional CNN -> Bi-directional LSTM (to capture context) -> Attention over B-LSTM to decode characters

- Attention Mask: can tell which part of the input corresponded with maximal output
    - [[Donahue et al., CVPR 2015]](https://arxiv.org/abs/1411.4389)

### CONCLUSION

- RNNs solve the problem of variable length input and output

- Solves knowledge of previous unit (by passing state)

- Can be trained end-to-end

- Finds alignment between input and outputs (through attention also)
