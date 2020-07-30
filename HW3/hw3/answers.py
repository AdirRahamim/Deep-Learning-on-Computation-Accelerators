r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0, seq_len=0,
        h_dim=0, n_layers=0, dropout=0,
        learn_rate=0.0, lr_sched_factor=0.0, lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=256, seq_len=64,
        h_dim=128, n_layers=3, dropout=0.3,
        learn_rate=0.01, lr_sched_factor=0.5, lr_sched_patience=2
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "Hello"
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
* First, for large corpuses we might have memory problem if we not split it to smaller sequences.
* Second, by splitting the corpus we try to avoid the model from overfitting and memorize the corpus, we create 
diversity in the learning process and let the model try to learn various data and not only one. We can expect that
sentences that are far away will have different meaning and thus it will learn various of sentences with different
meaning and thus generalize better.
* Third, it speeds up the training process as calculations on small sequences are faster than on the whole corpus.
"""

part1_q2 = r"""
**Your answer:**
There are the hidden layers that pass forward all the previous learned information data, thus enabling memorize
more than the sequence length.
"""

part1_q3 = r"""
**Your answer:**
We want the network to learn the logic and continuity of the text, if we shuffle it the sentences will be mixed and the
hidden state the network will pass to the next level will not have context, and the network will not learn meaningful 
and structured sentences.
"""

part1_q4 = r"""
**Your answer:**
1. Higher temperature will cause the model to be less confident and explore new options. For training this is a good 
practice and we probably will achieve better learning. Lower temperature will choose with more confident the higher
probability option, and on sampling we want it to be less random and more structured.
2. When the temperature is too high the probabilities are more uniform, thus making the model peak letters almost randomly 
and the output text we recive seems to have no meaning.
3. When the temperature is too low the model repeats the same words over and over again.
Low temperature makes the model to be too much confident in itself, the higgher probability gets very high and we will
choose it vocer and over again.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 8
    hypers['h_dim'] = 32
    hypers['z_dim'] = 8
    hypers['x_sigma2'] = 0.0007
    hypers['learn_rate'] = 0.00033
    hypers['betas'] = (0.9, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
The $\sigma^2$ hyper parameter control how much weight we wish to put on the reconstruction error of the model in
the loss function(in an inverse manner).
For low $\sigma^2$ values, we would expect more weight would be on the reconstruction loss, and thus the model will
try to do its best on make the output as much as closer to the input images.
For high $\sigma^2$ values, we would expect more weight will be on the KL-div loss, and we get the output images tends
to be very similar to each other.
"""

part2_q2 = r"""
**Your answer:**
1. The purpose on the VAE reconstruction loss is to increase the probability that the reocinstructed image will be 
close to the original image, in other words it tries to make the reconstructed image as much as close to the original 
one.
The KL divergence between two probability distributions measures how much they diverge from each other. Minimizing the 
KL divergence means optimizing the probability distribution parameters (μ and σ) to closely resemble that of the target 
distribution. We want that the encoder outputs will try to fit the distribution of the posterior distribution.
2.If we put more weight on the KL divergence loss we will get the the distribution of the encoder will be close to
standard normal distribution.
3. By using the KL loss term we reduce overfitting the model. Without using this term we might get sparse distribution
with null spaces(areas the encoder and decoder never seen), thus we will geet only reconstruction model without the
ability to work with unseen data(aka overfitting).
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=32, z_dim=128,
        data_label=1, label_noise=0.28,
        discriminator_optimizer=dict(
            type='Adam',
            weight_decay=0.02,
            betas=(0.5, 0.999),
            lr=0.0002,
        ),
        generator_optimizer=dict(
            type='Adam',
            weight_decay=0.02,
            betas=(0.5, 0.999),
            lr=0.00021,
        ),
    )
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
In the train_batch function, in the part where we train the discriminator we discard the gradients when we sample from 
the GAN, the reason for discarding the gradients in the discriminator training is that we dont want the discriminator 
training to affect the generator training.
The GAN is built out of two models - discriminator and generator, and as so we also split the training process and we 
dont want the one model training will affect the other one.
We want only the generator gradients to be affected only by it's loss, and to not update during discriminator train. 
"""

part3_q2 = r"""
**Your answer:**
1. No. The GAN is built out of two models, the generator and discriminator, and it's success depends on the two pars.
Low generator loss dosent mean good discriminator, it might mean that we have poor discriminator and thus the generator
can fool him, therefore we cant stop the training process based on some threshold for the generator loss.
2. Low generator loss means it gets better in fooling the dicriminator, but if also the discriminator loss
stays constant it means it also get better at identifying the true images, but still the generator becomes better at
fooling it.
"""

part3_q3 = r"""
**Your answer:**
The results are differ in their noiss level, where the VAE generated images are much more clearer and smooth, in all
the training process the images are getting more bright in each step, where in the GAN the images have a lot of noise שמג 
they are not sharp.
The difference is related to the training process of the methods, where the VAE has a final target that it try to fit it
, the GAN training process is more complicated, with two models that one try to trick the other and the generator
see every noise as a feature to distruct the discriminator, thus result in lower quality images.

"""

# ==============


