# Meaning-preserving Continual Learning

This is a follow-up to [domain_IL](https://github.com/rom1mouret/domain_IL).
The core idea remains the same.

### Introduction with an example

Your system's training journey might start like that:
- task A1: time T0 to T1: training the model to recognize human faces.
- task B: time T2 to T3: training the model to tell cats and dogs apart.
- task A2: time T4 to T5: training the model to get better at recognizing human
faces, maybe in a novel environment.

At time T1, your model and its latent units are perfectly fit for recognizing
faces within domains it was trained on.

At time T2, your model moves on to learning to distinguish between cats and dogs.
It can build on task A1's latent layer but it should not interfere with it
because the function of task A1's latent layer is to recognize human faces, not
to recognize human faces AND pets. In programming terms, task A1's network is
entirely frozen between T2 and T3.

At T4, we are back to face recognition.
Task A1's latent representation can be safely refined as long as it keeps on
fulfilling the same invariant function.
This might interfere with Task B, but not in a
[destructive manner](https://en.wikipedia.org/wiki/Catastrophic_interference).

### Theoretical framework

In broad terms, we anchor/tether/ground latent representations by forcing them
to remain good at realizing the function they were originally trained for.
How the system realizes the
function-that-latent-representations-were-originally-trained-for *is* what
defines latent representations.

An important distinction to make is that such functions are not about mapping
some input domain to an output domain.
It's all about the output.
Recognizing faces in photographs fulfills the same function as recognizing faces
in real life.
The input domain hardly matters `*`.
Your friend's face is an abstraction in virtue of the realizability of the same
recognition function across many domains/environments.

Now, my goal is to formalize this idea and characterize the "meaning of latent
units" in more rigorous terms.

[link to MPCL Framework v1 pdf](MPCL-Framework-v1.pdf)

I will also attempt to frame bare-bones [Domain-IL and Class-IL](https://arxiv.org/pdf/1904.07734v1.pdf)
within this framework (see below).
Admittedly, it takes a bit of shoehorning because simple systems don't create
many opportunities for MPCL to take advantage of the complexity to find
inconsistencies, for example by cross-checking predictions from multiple modules
of the system.  
Moreover, complex systems are generally upheld by high-dimensional latent
representations, with a [lot of empty space](https://en.wikipedia.org/wiki/Curse_of_dimensionality#Blessing_of_dimensionality)
wherein you would typically spot inconsistent configurations of latent values.
MPCL relies on finding inconsistencies to detect domain boundaries.

`*` At the very least, *some* of the latent units share the same function
between photographs and real life, while *some* other units' function might be
specialized for specific kinds of domains.


### Terminology

A *group* of latent units has a *function*.
Within this group, individual latent units have a *meaning*.

It's called the *Meaning*-preserving Continual Learning framework because I plan
on extending this idea to function-free intrinsic meaning (latent units that get
their meaning from adjacent units), in contrast to extrinsic meaning (latent
units that get their meaning from external labels/feedback). "Function" wouldn't
be a good fit here.

Also, the typical way of detecting inconsistencies is to look for configurations
of latent values that do not realize any function, i.e. they are meaningless.
In this context, "meaning" is intuitive.

## Domain-IL classification on Permuted MNIST

In this scenario, the most straight-forward approach is to tether the latent
units directly to the output classes.
To do so, we train a classifier to map latent units to classes on an arbitrary
domain and freeze the classifier right away.

Following the naming conventions in [MPCL-Framework-v1](MPCL-Framework-v1.pdf),
*C* is a softmax classifier and *e* is negative cross-entropy.

Next, we train a processor for each domain under the constraint that they must
be good at predicting the target labels.

The tricky part is to detect domain boundaries when making predictions. We have
to resort to using a surrogate function.

Here are the results on [Permuted MNIST](https://arxiv.org/pdf/1312.6211.pdf),
using the confidence of *C* as a surrogate for the degree to which the output
of a processor "conveys meaning".

![Permuted MNIST](images/mnist_results.png)

Multiple rounds are plotted. The average is represented with the most opaque red.

code: [mnist_domain_il.py](mnist_domain_il.py)

## Class-IL on EMNIST

For Class-IL, I chose EMNIST because it includes more classes than MNIST.
The anchoring step is performed on EMNIST digits while the continual learning
and the evaluation are done on EMNIST letters.

MPCL framework doesn't lend itself well to Class-IL, but with a bit of trickery,
EMNIST Class-IL can be expressed as a Domain-IL problem.

- instead of training a classifier to recognize characters, have it predict how
characters are rotated or flipped. This will anchor the latent units just as
well as letter recognition.
- interpret each letter as a distinct domain.

The training algorithm is essentially a copy-paste of MNIST Domain-IL except
that we randomly rotate/flip the training digits on the fly.

#### predicting

All the 7 transformations (90-rotation, horizontal flip etc.) are controlled by
us. Therefore, the expected output of the classifier is always known, even on
the test set.

To make predictions, we apply every transformation to each letter and choose the
domain for which the classifier is the most correct regarding how the letter has
been transformed.

![Permuted MNIST](images/emnist_results.png)

code: [emnist_class_il.py](emnist_class_il.py), [plot_emnist.py](plot_emnist.py)

The green curve shows how a collection of one-class classifiers would perform,
under some simplifying assumptions.

- false positive rate is 5%, in the same ballpark as my classifier.
- false negative rate is 0%.
- if multiple classifiers report a positive match, we randomly choose between them.


## Triangular activation

The layers of the processors are activated by an unusual function.

```python3
class Triangle(torch.nn.Module):
    def forward(self, x):
        return x.abs()
```

With ReLU and the like, out-of-distribution inputs saturate the activation
functions almost as much as in-distribution inputs, which makes ReLU networks
ill-equipped for measuring confidence levels on OOD data.

It is not a matter of calibration.
It's fine for a classifier to be overconfident / underconfident in a consistent
fashion, but we can't afford the network to throw in the towel when it
encounters something it doesn't know, like noise.  

I prefer `Triangle` because it doesn't throw anything away. It merely folds the
input space.

Disclaimer: I haven't studied in depth the dynamics of Triangle-activated
networks so it's possible that Triangle doesn't do what I think it does.
I have noticed improvements on Domain-IL though.

## Softmax

Softmaxing the last layer of the classifier was a mistake and I will try without
softmax in subsequent experiments.
Softmax is not a good choice for MPCL because there are infinite solutions to
`softmax(x) = y`, thus it doesn't constrain `x` enough for `x` (or anything
upstream of `x`) to be transferable to other tasks.

It doesn't affect Permuted MNIST and EMNIST results that much because latent
representations are not transferred to other tasks in our experimental setup.
