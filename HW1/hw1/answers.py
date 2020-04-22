r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Increasing K up to some point can lead to better results(In this particular example best K I obtained was K=3) since 
we take to account more neighbors and those protect our results from exceptions. However if we look at the extremal values of K:
for K=1 the result is obtained by the closest neighbor, in this case the model is overffiting  and we won't get
optimal results since the model might be sensitive to noises.
On the other extremal, if K=#(of samples) we most likely to increase the probability to include wrong images in our evaluation,
and for very large K, we might get more images than the total number of matching images in the testing set, thus include
wrong images for sure.
And indeed, the best value of K obtained for K=3.**

"""

part2_q2 = r"""
**We use K-fold CV in order to avoid over fitting and try to generalize our model.
1)If we select the best model with respect to train-set accuracy we will over fitting the model parameters according to
the training set instead of try to generalize them, thus get poor performance.

2)In this scenario we will get the model over-fitted to the test images. The test purpose is to evaluate the results
of the model on unseen data, and by training on the test-set we will overffit the model on unseen data and thus
can expect to get bad results on unseen data.
**

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**The rule of $\Delta$ is to make sure that the score of the correct class is higher than the other class by at least 
some margin $\Delta$. Thus $\Delta$ has to be positive, but it dosen't matter what exact value $\Delta$ will get, since
the weights can be scaled up or down accordingly in the training process. 
**
"""

part3_q2 = r"""
**1) The model is trying to find a linear plane that separates between the different classes, in our case 0-9 digits.
We can see that letters that are more close to each other, for example 5 and 6, have more visually alike weights matrices,
And those the model mistakes when he faced letters that are different from general example and look very similar to 
another digit, and thus because their weight matrices close to each other he fails in that test.

2)The KNN model looks at the K most similar images and by that he judges which class the image belong to, 
where as SVM approach is to separate between whole classes and judge which class the image belongs to by taking the 
highest score class.
**

"""

part3_q3 = r"""
**1)I think the learning rate I chose was good.
If the learning rate was too low, it wouldnt converge at all, but we can clearly see that after small amount of epochs
the loss got extremly low, thus the learning rate is not low.
If the learning rate was to high we would see spikes in the graph, the loss will go down and in some point go up again 
as we would pass the minimum point.
Thus the learning rate is good.

2)The model is slightly overfitted to the training set. There is a constant gap in the accuracy between the training
set and the validation set, in favor of the training set, thus the model is overfitted to the training set,
howewver the gap is very small and constant(it is not growing), thus it is slightly overfitted.
**

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**The ideal case would be if the residual would be zero, i.e $y - \hat{y} = 0$ and in the graph we will see that all the
dots are on the y=0 line. We can clearly see that using CV gave us better results as the MSE got lower and the dots
are closer to y-0 line.
**

"""

part4_q2 = r"""
**1)The affect of W in the regularization term is higher than the affect of $\lambda$ on it, so we can assume that small
changes in $\lambda$ wont change the loss dramatically. Thus we use logspace in order to check different order
of magnitudes and enable to test wide scales in small amount of tests, instead of much higher test number we needed
to use by linespace.

2)Number if times we fitted the model to data is: $$#num_lambda_valus * #number_degrees * #num_k_foldds = 20 * 3 * 3 = 180$$
**

"""

# ==============
