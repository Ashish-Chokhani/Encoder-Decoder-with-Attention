# Author: Ashish Chokhani
--- 

 this

- History: Fukushima (1980), LeNet (1998), Many-layered MLP with BackProp (tried, but without much success and with vanishing gradients), Relatively recent work

### [ALEXNET, Krizhevskky et al, 2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - Winner of ImageNet 2012
    - Trained over 1.2M images using SGD with regularization
    - Deep architecture (60M parameters)
    - Optimized GPU implementation (cuda-convnet)




- Mini-Batch SGD: Compute the gradients for a mini-batch of training examples drawn randomly, update the parameters

#### Advantages of SGD over Batch GD
    - Usually much faster than Batch GD
    - Noise can help! There are examples where SGD produced better results than Batch GD

#### Disadvantage of SGD
    - Can lead to no convergence! Can be controlled using learning rate though

### FIRST-ORDER OPTIMIZATION METHODS

### Momentum
    - Momentum == Previous Gradient
    - Can increase speed when the cost surface is highly non-spherical
    - Damps step sizes along directions of high curvature, yielding a larger effective learning rate along directions of low curvature
    - Can help in escaping saddle points… But usually need second order methods to escape saddles

### Accelerated Momentum

### Nesterov Momentum
    - Compute the Gradient Delta from loss at the weights updated by the previous gradient delta, then combine this Gradient Delta with the previous gradient delta
    - Somehow works better than just momentum

### RMSprop

### AdaGrad
   - Gradients diminish faster here… RMSprop takes care of this

### Adam
    - Similar to RMSprop with momentum

### AdaDelta
   - Works well in some settings

- Vanilla SGD may get stuck at saddle points

- Many papers use vanilla SGD without momentum, but best bet is Adam

- ![alt text](http://ruder.io/content/images/2016/09/contours_evaluation_optimizers.gif “Sebastian Ruder’s”)

### SECOND-ORDER OPTIMIZATION METHODS

- Gradient descent comes from approximating the Taylor series expansion to just the first order term

 - By considering the second-order terms:

$${\delta}w = -\frac{f’(x_{0}}{f’’(x_{0})}$$

- So we see that the gradient update is inversely proportional to the second derivative!

- So, gradient update = $-H^{-1} * g$, where $H$ is the Hessian, $g$ is the gradient

- The ideal learning rates are the Eigenvalues of the Hessian matrix

- But, the Hessian is not feasible to compute at every step, let alone its inverse and eigenvalues
    - Computational burden
    - Cost-performance tradeoff is not attractive
    - There are some Newton’s methods to try, but they are attracted towards saddle points

- So we use other approximations like the Quasi-Newton methods
    - We compute one Hessian, then we compute the difference to add to get the next Hessian
    - E.g. LGFBS
    - Approximations: Fischer Information, Diagonal approximation

- Generally, second-order methods are computationally intensive, first-order methods work just fine!

### RECENT ADVANCES

### Gradient Descent

- Trying to understand the error surfaces, escaping saddles, convergence of gradient-based methods

- [Deep Learning without Poor Local Minima, NIPS 2016](https://arxiv.org/abs/1605.07110)
    - Theoretically showed that converging to a local minimum is as good as global minimum

- [Entropy-SGD: Biasing Gradient Descent into Wide Valleys, ICLR 2017](https://arxiv.org/abs/1611.01838)
   - Try to push gradient descent towards flat minima, because it is those minima into which most networks now converge to anyway

### Escaping Saddle Points

- [Escaping from Saddle Points, COLT 2015](http://proceedings.mlr.press/v40/Ge15.html)
    - Add Gaussian noise to gradients before update

- Degenerate Saddles
    - We found that most networks actually converge to saddles, not even flat minima
    - Saddles may be good enough

### REGULARIZATION

- Difference between ML and Optimization is, ML tries to generalize, by designing an optimization problem first

- Theoretical ML: Empirical Risk Minimization (ERM). But, ERL can lead to overfitting

- Avoiding overfitting is Regularization

- It’s not important to get 0 training error, it’s more important to find the underlying general rule

### Early Stopping
    - Stop at an earlier iteration before training error goes to 0
    - Of course not a very good idea
    - Error-change criterion: if error hasn’t changed much within n epochs, stop
    - Weight-change criterion: if change in max pair-wise weight difference hasn’t changed much, stop
        - Use max, it might be that only one direction is learning while all others are not

### Weight Decay
    - In addition to prediction loss, also add a loss with the magnitude (2-norm) of the weights
    - Don’t give me a very complicated network
    - Using L1-norm weight decay gives sparse solutions, meaning many weights go to 0, which helps in compression

### DropOut
    - Does model averaging
    - In training phase: randomly drop a set of nodes in hidden layer, with probability p
    - In test phase: use all activations, but multiply them with p
        - This is equivalent to taking the geometric mean of all the models if you ran them separately
    - This is equivalent to using an ensemble
    - Dropping nodes helps nodes in not getting specialized
    - [Srivastava et al., JMLR 2014](http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf)

### Regularization through Gradient Noise
    - Simple idea: add noise to gradient
    - [Neelakantan et al., Adding gradient noise improves learning for very deep networks, 2015](https://arxiv.org/abs/1511.06807)

### DATA MANIPULATION METHODS

### Shuffling inputs
    - [Efficient backprop, LeCun](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

### [Curriculum Learning [Bengio et al., ICML 2009]](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf)
    - Old idea, proposed by Elman in 1993
    - Use simpler training examples first

### Data Augmentation
    - Data jittering, rotations, color changes, noise injection, mirroring
    - [Deep image: Scaling up image recognition [Wu et al., 2015]](https://arxiv.org/abs/1501.02876)

### Data Transformation
    - Normalize/standardize the inputs to have 0 mean and 1 variance (refer to [Efficient Backprop paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf))
    - Decorrelate the inputs (probably using PCA)

### Batch Normalization
    - DropOut fell out of favour after this was introduced
    - Introduced Internal Covariance Shift
    - Normalize the activation of a neuron across its activations for the whole mini-batch; then multiply with a variance and add a shift, and learn those mean and variance parameters in training
    - BatchNorm eliminated the need for good weight initialization
    - Also helps in regularization, by manipulating parameters with the statistics of the mini-batch

### ACTIVATION FUNCTIONS

- Sigmoid: Historically awesome, but gradients fall to 0 if activations fall beyond a small range

- tanh: Same problem

- ReLU: created to counter the zero-gradient problem
    - But, ReLU is not differentiable
    - So we assume ReLU’s gradient as a sub-gradient: we define the gradient at 0
    - Dying ReLU problem: If even once the activation goes to negative, the ReLU does not let any neuron before it learn anything during backprop

- LeakyReLU: created to counter the Dying ReLU Problem

- Parametric ReLU: Same as LeakyReLU, but a different slope for negative values than LeakyReLU

- MaxOut: generalization of ReLU
    - Take the max of a bunch of network activations

- Try tanh, but expect it to work worse than ReLU/MaxOut

### LABELS

- Try smooth labels (0.9 instead of 1)

### LOSS FUNCTIONS

- Cross-entropy (will simplify to similar to MSE)

- Negative Log-Likelihood (will simplify to similar to CE)

- Classification Losses, Regression Losses, Similarity Losses

### WEIGHT INITIALIZATION

- Don’t initialize with 0, no backpropagation, no learning

- Don’t initialize completely randomly, non-homogeneous networks are bad

- Most recommended: [Xavier Init [Xavier Glorot, and Yoshua Bengio, Understanding the difficulty of training deep networks, AISTATS 2010]](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
    - Uses fan_in + fan_out info to draw samples from a uniform distribution

- Caffe made a simpler implementation

- He made a better implementation because Caffe’s didn’t work for ReLU: [Delving deep into rectifiers, [He et al.]](https://arxiv.org/abs/1502.01852)

- RNNs

- Backpropagation through time

- Vanishing gradient problem

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

### Scheduled Sampling
- During testing, it might just happen that the RNN gives one wrong output, and the error is compounded with time since the wrong output is fed as the next input!

- Scheduled Sampling is employed to take care of this

- During training, from time to time, sample from the output of the RNN itself and feed that to the next decoder input instead of the correct input

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

### SAMPLING/GENERATION

- Use “tau” as temperature to modify the output probabilities: s = s/tau
    - tau = 0 => prob is infinity for one word
    - tau = infinity => prob is flat, so you might not have trained your RNN at all

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

- Problem to solve: unsupervised learning of (visual) representations

- Given access to a lot of data, we want to generate the data. Generation => We have understood the data.

- Initially, we wanted to learn some properties of data by trying to reconstruct the data using a deep network
    - Unlabelled Data -> Deep Network -> SGD Objective with Reconstruction Loss
    - But, this network simply formed the identity function w.r.t. the input data.
    - Didn’t at all work for data other than input data
    - Deep learning is a bad student who just memorises whatever is given to him, but doesn’t understand anything about the data

- Then, [Hinton & Salakhutdinov, in Science 2006](https://www.cs.toronto.edu/~hinton/science.pdf) made an autoencoder, where they put a bottleneck in the middle of the deep network
    - This worked well, the network learnt a compressed version of the input data
    - But it wasn’t there yet, it just learnt a fancy compression

- [Denoising Autoencoders [Vincent et a;., ICML 2008]](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)
    - Give purposefully corrupted data -> Autoencoder -> Reconstruct the original image without noise

- [Sparse Autoencoder: Building High-level Features Using Large Scale Unsupervised Learning, ICML 2012](https://arxiv.org/abs/1112.6209)
    - Used 16000 processors, 1 billion connections, 10M images to solve the problem of unsupervised learning once and for all
    - Worked mostly for cat videos. Google argued that that’s because YouTube is full of cat videos

- GANs work! How do we know? After all, Stacked Autoencoders also went only so far... Results!
    - [BEGAN: Boundary Equilibrium GAN](https://arxiv.org/abs/1703.10717): Uses WGAN + encoders

- GANs: Counterfeit game analogy

- GAN Architecture [Ian Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661)
    - z is some random noise = a latent representation of the images
    - GAN is not trying to reconstruct the input data
    - GAN is trying to make a model such that it generates data that is indistinguishable from real data
    - Generator vs Discriminator
    - G and D must be trained with complementary backprop gradients

- [GAN Tutorial, NIPS 2016](https://arxiv.org/pdf/1701.00160.pdf)

- G tries to:
    - Maximise D(G(z)), or minimise (1 - D(G(z)))
    - Minimise D(x)
    - => Minimise D(x) + (1 - D(G(z)))

- D tries to:
    - Maximise D(x)
    - Minimise D(G(z)), or maximise (1 - D(G(z)))
    - => Maximise D(x) + (1 - D(G(z)))

- Thus, GAN objective function:

$$min_G max_D [D(x) + (1 - D(G(z)))]$$

- To analyze average behaviour, we take expectation:
$$ min_G max_D E[D(x)] + E[(1 - D(G(z)))]

- At equilibrium, G is able to generate real images, and D is thoroughly confused

- Minimax Game:

$$ J^{(D)} = - {\frac{1}{2}}E_{x~p_{data}} log D(x) - {\frac{1}{2}}E_{z} log (1 - D(G(z)) $$
$$ J^{(G)} =  - {\frac{1}{2}}E_{z} log D(G(z) $$

- But Goodfellow’s paper didn’t have good enough results… Next good progress was using DCGANs

- [Unsupervised Representation Learning with DCGAN, Radford et al., ICLR 2016](https://arxiv.org/abs/1511.06434)
    - Used a Deep Deconvolutional Network as the generator: Removed fc layers, changed activations, used BatchNorm
    - First realistic generation of images. Autoencoder, etc. produced blurry and erroneous images, this one made good sharp images
    - Also, latent vectors capture semantics as well, like <Man with glasses> - <Man without glasses> + <Woman without glasses> = <Woman with glasses>

- [GAN Hacks by Soumith Chintala](https://github.com/soumith/ganhacks)

- [Data-dependent initialization](https://arxiv.org/abs/1511.06856); GitHub: [magic_init](https://github.com/philkr/magic_init)
    - To change pre-trained weights according to new data format (-1 to 1 instead of 0 to 1, etc.)

### Applications:

- Image from Text

- 3D Shape Modelling using GANs

- Image Translation: Pix2Pix
    - But this uses pairs of images
- [Next Video Frame Prediction [Lotter et al., 2016]](https://arxiv.org/abs/1605.08104)


### [Unrolled GANs](https://arxiv.org/abs/1611.02163)

- Based on letting the generator “peek ahead” at how the discriminator would see

- Does much better than DCGANs

### AUTOENCODERS

- Linear AE is same as PCA

- But fails:
    - Just memorizes the data, doesn’t understand the data
    - Overcomplete case: hidden dimension h has higher dimensions
    - Problem: identity mapping

- Sparsity as a regularizer

- De-noising autoencoder [Vincent et al.]

- Contractive Autoencoder
    - Learns tangent space of manifold
    - Minimize Reconstruction error + Analytic Contractive loss
    - Tries to model major variation in the data

- Can we generate samples from manifold?
    - We need the lower-dimensional distribution
    - Let’s assume it is Gaussian

### VARIATIONAL AUTOENCODER

- Decoder samples z from a latent space, and decodes into the mean and variance from which to sample an x

- Encode: x -> Encode to a mean and variance for z, Decode: Sample a z -> Decode to a mean and variance for each dimensional value of x

- Terms:
    - Posterior: $p(z | x)$
    - Likelihood: $p(x | z)$
    - Prior: $p(z)$
    - Marginal: ${\int}p(x, z) dz$

- But, the marginal is intractable, because z is too large a space to integrate over

- So, decoder distribution p(x | z) should be maximized over q(z | x): minimize the log likelihood $E_{q(z | x)}[log p(x | z)]$

- Also, q(z | x) should be close to the z prior p(z) (Gaussian): minimize the KL Divergence $KL[q(z | x) || p(z)]$

- Importance Sampling
    - But, intractable

- Jensen’s inequality

** Concave: Graph is above the chord

- Variational Inference
    - Variational lower bound for approximating marginal likelihood
    - Integration is now optimization: Reconstruction loss + Penalty

### VARIATIONAL AUTO-ENCODER

- Latent variable mapping [Aaron Courville, Deep Learning Summer School 2015]
    - Trying to discover the manifold in the higher-dimensional space, trying to find the lower-dimensional numbers

- Example of Face Pose in x axis and Expression in y axis from the Frey Face dataset

- We can sample from z and generate an x using a transformation G: x = G(z)

- How to get z, given the x’s in our dataset

- [Auto-Encoding Variational Bayes, Kingma and Welling, ICLR 2014]

- Estimate ${\theta}$ without access to latent states

- PRIOR: Assume Prior $p_{\theta}(z)$ is a unit Gaussian

- CONDITIONAL: Assume $p_{\theta}(x | z)$ is a diagonal Gaussian, predict mean and variance with neural net

- So, from a sampled z, estimate a mean and variance from which to sample an x

- From Bayes’ theorem,

$$ p_{\theta}(z | x) = {\frac{p_{\theta}(x | z) * p_{\theta}(z)}{p_{]theta}(x)}} $$

- Here, p_{\theta}(x | z) can be found from the decoder network,  p_{\theta}(z) is assumed to be a Gaussian, BUT $p_{]theta}(x)$ is intractable. 
    - Because, to find $p_{]theta}(x)$, we need to integrate over all x’s on all values of z

- So, $p_{\theta}(z | x)$ is very hard to find, since we don’t know $p_{]theta}(x)$

- Instead, we approximate this posterior with a new posterior called $q_{\phi}(z | x)$, and then try to minimize the KL divergence between $q_{\phi}(z | x)$ and $p_{\theta}(z | x)

- We get $q_{\phi}(z | x)$ from the encoder network.

- Data point x -> Encoder: mean, (diagonal) covariance of $q_{\phi}(z | x)$ -> Sample $z$ from $q_{\phi}(z | x)$ -> Decoder: mean, (diagonal) covariance of $p_{\theta}(x | z)$ -> Sample $hat{x}$ from $p_{\theta}(z | x)$

- Reconstruction loss for $hat{x}$, Regularization loss on prior  $p_{\theta}(x)

- Reparametrization trick
    - Because we’re sampling in between, it is not back-propagatable, since we need to differentiate through the sampling process

- [Laurent Dinh Vincent Dumoulin’s presentation](http://www.iro.umontreal.ca/~bengioy/cifar/NCAP2014-summerschool/slides/Laurent_dinh_cifar_presentation.pdf)

### Semi-Supervised VAEs
    - M1 - vanilla VAE: use z to generate x
    - M2 - use z and a class variable y to generate x
    - M2 - use some inputs with labels and some without
    - M1 + M2
        - shows dramatic improvement
    - Conditional generation using M2

### Conditional VAE
    - z does not split the different classes, instead class label is given as a separate input

### [Importance-Weighted VAE, Salakhutdinov’s group, ICLR 2016](https://arxiv.org/abs/1509.00519)

### De-noising VAE

- Added noise to input

- Took $q_{\phi}$ as a mixture of Gaussians

### Applications
    - Image and video generation
    - Super-resolution
    - Forecasting from static images
    - Inpainting

### AUTOREGRESSIVE MODELS

### [RIDE by Theis et al.](https://arxiv.org/pdf/1506.03478.pdf)

- Recurrent Image Density Estimator

- Structure: SLSTM + MCGSM

- Spatial LSTM: Kind of a 2D LSTM

- Mixture of Conditional Gaussian Scale Mixtures

### [Pixel RNN by Oord et al.](https://arxiv.org/abs/1601.06759)

- Improvisation over sLSTM

- Uses two different LSTM architectures: RowLSTM, Diagonal BiLSTM

### [Pixel CNN](https://arxiv.org/abs/1606.05328)

- Replace RNNs with CNNs

- Can’t see beyond a point, but much faster to train

### Conditional Pixel CNN

### [PixelCNN++](https://arxiv.org/abs/1701.05517)

- PixelCNN produces discrete distribution, this one produces continuous distribution

### DISCRIMINATIVE MODELS

- Image de-noising, image de-blurring, image super-resolution

- Con: there has to be a separate model each type of degradation

- Degraded image: Y = A*X + n

- Generative modelling: versatile and general

- MAP inference: arg max_X p(Y | X)*p(X)

- Traditional approach: Dictionary learning
    - [Video dictionary [Hitomi et al., 2011]](http://www.cs.toronto.edu/~kyros/courses/2530/papers/Lecture-10/Hitomi2011.pdf)
    - But, difficult to capture long-range dependencies

- GANs are out of the question, since they don’t model p(x) explicitly. MAP inference is not possible.

- VAE takes a lot of computation: they approximate p(x) with a lower bound

## [Mitesh Khapra](http://www.cse.iitm.ac.in/~miteshk/), IIT Madras - Deep Learning for Word Embeddings

- “You shall know a word by the company it keeps” - Firth., J. R. 1957:11

- In One-Hot Encoding, every pair of points is sq(2) Euclidean distance between them. Using any distance metric, every pair of words is equally distant from each other. But we want an embedding that captures the inherent contextual similarity between words

- A Co-occurrence Matrix is a terms x terms matrix which captures the number of times a term appears in the context of another term
    - The context is defined as a window of k words around the terms
    - This is also called a word x content matrix

- Stop words will have high frequency in the Co-occurrence Matrix without offering anything in terms of context. Remove them.

- Instead of counts, use Point-wise Mutual Information:
    - PMI(w, c) = log(p(c | w)/p(c)) = log(count(w, c)*N/(count(c)*count(w)))
    - So Mutual Information is low when both words occur quite frequently in the corpus but don’t appear together very frequently
    - PMI = 0 is a problem. So, only consider Positive PMI (PPMI): PPMI=PMI when PMI>0, PPMI=0 else

- It’s still very high-dimensional and sparse. Use PCA:
    - SVD: X_{mxn} = U_{mxk}{\Sigma}_{kxk}V^T_{kxn}, where k is the rank of the matrix X
    - Make k = 1, or any number lesser than the rank of X, and U*{\Sigma}*V^T is still an mxn matrix, but it is an approximation of the original X, wherein the vectors are projected along the most important dimensions, and it is no longer sparse

- X*X^T is the matrix of the cosine similarity between the words. X*X^T(i, j) captures the similarity between the i^{th} and j^{th} words. 

- But this is still high-dimensional. We want another approximation W, lesser dimensional than X, s.t. W*W^T gives me the same score as X*X^T
    - $X*X^T = (U{\Sigma}V^T)*(U{\Sigma}V^T)^T = (U{\Sigma}V^T)*{V{\Sigma}U^T} = U{\Sigma}*(U{\Sigma})^T$, because V is orthonormal (V*V^T = I).
    - So, U{\Sigma} is a good matrix to be our W, since it is low-dimensional (m x k).

- Iti pre-deep learning methods

### CONTINUOUS BAG OF WORDS (CBoW)

- Given a bag of n context words as the input to a neural network, predict the (n+1)^{th}word as the softmax output of the network.
