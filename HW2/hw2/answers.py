r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.2, 5e-3, 1e-3
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.1, 0.031, 0.0029, 0.00033, 0.00015
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd, lr = 0.1, 0.003
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Dropout is a technique used to avoid overfitting, thus we could expect the non-dropout graph will overfit the train 
data(high train accuracy - almost 100% !) and will have higher test-set loss than the dropout version
(and also lower test-set accuracy than the dropout version).
And indeed, we can see than the non-dropout version have higher test-set loss and average test-set accuracy(lower than 
dropout = 0.4), so the results most of the time much my expectations(except for test loss with dropout=0.8)
First, we can see dropout p=0.8 is too high, as it is far behind the other two, possibly we dropout too many neurons 
and the learning time is too low.
For dropout p=0.4 we can see that train-set accuracy is lower than non dropout, but test-set accuracy is the highest,
so a dropout value in the middle of 0.4 is a good choice for generalize the model.**
"""

part2_q2 = r"""
**Yes, this scenario is possible.
 The accuracy only measures the number of correct predictions(correct or wrong), i.e the maximum of of the class score 
 is the correct prediction, where the cross-entropy loss measures how "strong" the prediction is. 
 For example for a class with two labels and the following scores for the correct labels: {0.49,100}, we will get 
 accuracy of 50% and low loss, but if next epoch we get the next result: {0.51,0.51} we get accuracy of 100% -> accuracy
 increased, but the loss also increased as the true label score of the second label dramatically got lower.**

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**
(1)
With a very deep net we will expect it will be hard for the network to learn the data, and indeed with L=16 in our 
experiment, the network couldn't achieve any learning under our experiment conditions, because of vanishing gradient and
too small training set, the best test-set accuracy achieved in not too depth but not too small network - L=4, 
with about 10 epochs to converge.
(2)
For L=16 the network couldn't learn and stopped the training process in the beginning. The reason that we had no learning 
in the deep network is the known vanishing gradient problem, which make the gradient to zero or explode at some point.
Two possible additions to the network that can alleviate this problem are batch normalization and the add of skip
connections, as we learned both this methods are very helpful in training deep networks and overcome vanishing gradient.    
**
"""

part3_q2 = r"""
**
First, we can see that for all values of L the greater number of filters(K = 128/258) achieved the best results, and
in both tests configuration the narrow networks(with K=32) achieved the worst results.
We can explain this because it has more parameters to learn in each layer and thus can achieve better results.
For L=2: we can see that for K=128 best test-set accuracy achieved, better than K=32/64 values in Exp 1.1.
For L=4: we can see that for K=256 best test-set accuracy achieved, again, better than K=32/64 values in Exp 1.1.
For L=8: we can see that for K=128 best test-set accuracy achieved.
They all converged after about 10 epochs.
Compare to experiment 1.1: In general, the wider architectures, with K=128/256, achieved better results compare to the
thinner architectures in experiment 1.1, with K=32/64, as I explained above wider networks have more parameters to learn
in each layer.
**
"""

part3_q3 = r"""
**
The best test-set accuracy achieved for L=2. It seems that with various number of filters network the deeper network 
achieved a little less good results, perhaps because for deeper networks the learning takes more time.
We can see the using complicated network with various filters achieved 
better results in about 5% compared to experiment 1.2.
Converges took about 10 epochs.
**
"""

part3_q4 = r"""
**
We can see that the ResNet networks have better performance compared to the Conv classifier. 
In the first test, with K=32, again best accuracy achieved for the shallow network with L=8, and again for various number
of filter network, with K=[64-128-256], we got best accuracy not for the deepest network, but rather for the shallow
 network with L=2. It seems than in our tests the deeper networks learning is slower than the shallow networks.
In 1.1 experiment results for all values of L the results were pretty close with slight difference, here the difference 
is more noticeable, and as mentioned before, the results using ResNet are better compared to Exp 1.1.
Same as in experiment 1.3, we can see that the more wider network achieved better performance, but this time the best
results in both experiment is very close. 
**
"""

part3_q5 = r"""
**
(1)
I used the idea from the inception module and defined a custom convolution block, which was composed of 3 convolution 
with different kernels - 1,3 and 5, and the final output of the block was their weighted sum after batch normalization. 
Additionally, I used other techniques from ResNet such as skip connections, max pooling, batch normalization and 
dropout.
(2)
The best result in the experiment achieved for L=6 and L=3, again same as in experiment 1, shallow networks achieved 
better results. Converges took about 15 epochs.
Compared to experiment 1: In Exp.1.1 all different experiments were pretty similar, however here the difference is more
noticeable, in this experiment for L=16 we didnt had learning, here using all parameters we had learning,
and all the results are better compared to 1.1 results.
The results seem to be more stable compared to the experiments in Exp.1, the changes are less sharpen and we have smaller
glitches compared to Exp.1. 
**
"""
# ==============
