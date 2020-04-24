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
    wstd, lr, reg = 0.6, 5e-3, 1e-3
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.15, 5e-3, 5e-3, 1e-4, 1e-4
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd, lr = 0.1, 1e-4
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
"""

part2_q2 = r"""
*Your answer:*
Generally yes, this phenomena is indeed possible and could happen as the accuracy is binary and only measures the
percentage of correct lables, while the cross-entropy loss measures how "strongly" the prediction is (i.e. in terms of
the values of the softmax output).
Let us look at a case where some samples are now being classified correctly and cause an increase in accuracy
and in turn a decrease in loss, while in parallel, a big chuck of data is still being labeled the same (supposedly even
correct) label, but with lss confidence (e.g. predicting a true label of a cow but with 0.6% instead of 0.9% like it was
before). The latter would cause an increae in loss while not effecting accuracy. Hence, together the two could potentially
cause an increase in accuracy together with an increase in loss.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
*Your answer:*
1. Regarding the network depth, to our understanding, too little layers overfitted the data pretty quickly (wasn't expressive enough), while having too many layer didn't even manage to learn (we'll discuss this later), and the sweet spot in this configuration was 4 layers which proved better then the 8 layers in both K configurations (32, 64).
2. For 16 layers the network didn't suceed in training, to our understanding due to the amount of max-pooling, shrinking the input to an almost non-diffrentiable vector, not allowing the network to distinguish between the different inputs (hence converging to a the naive 10% random guess in a 10 class data set). This could be avoided either by reducing the number of max-pooling layers or adding up-scaling layers to keep the spatial diversity.

"""

part3_q2 = r"""
*Your answer:*
For shallower networks (2) using less filters proved better (hence a thinner configuration) but overall not a very sucessful one in terms of accuracy, while when making the network deeper, adding more filters improved the network and all in all for both the 8 and 4 layers, 128 feature maps seemed to be the sweet point in accuracy (while not too far from the 256). Comparing to section 1.1, these wider architectures proved better then the thinner ones we used in section 1.1.

"""

part3_q3 = r"""
*Your answer:*

In these set of tests we tested the same K block and for different network depths. In the very deep configuration we saw the same phenomena as in section 1.1 with the deepest configuration and other than that the middle configuration, with 9 layers total, allowed the network to achive best results before starting to over-fit. the 6 layer one overfitted the data pretty quickly and overall this configurations weren't as good as the ones in section 1.2 possibly because of the "saw tooth" type of block, increasing the filters number to 256 and then dropping back to 64 and again raising to 256 (but this is purely a speculation).

"""

part3_q4 = r"""
*Your answer:*
First off, the results for the resnet architecture were all together better then the previous configurations, and it also allowed larger networks to perform better (maybe due to its ability to maintain a memory of the original input over several layer with the skip layer), that said, we also saw it took it longer to train (makes sense).

"""

part3_q5 = r"""
*Your answer:*
1. In our architecture, and honestly without too much intuition, we mainly tried to reduce the number of max pooling to allow deeper networks (hence we max-pooled every other block) and also added a decent amount of dropout to allow more generalization. Other than that we kept the resnet block the same and the fc layers as before. We also tried playing around with different amounts of dropout but without any conclusive results.

2. Overall, with the very deep architecture we still didn't manage to perform better than before, but with the 18-layer configuration we did manage to cross the section 1.4 resnet results slightly and if we had more time we think playing with hyperparameters and increasing the early-stopping threshold a bit we could do even better.

"""
# ==============
