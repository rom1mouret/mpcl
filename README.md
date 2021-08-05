# Meaning-preserving Continual Learning (MPCL)

This is a follow-up to [domain_IL](https://github.com/rom1mouret/domain_IL).
The core idea remains the same but it is framed differently.

```diff
+ 2021 Feb update: MPCL rules are now explained in the slides.
+ 2021 March update: the slides now introduce the problem with intuition pumps.
```
Link to [MPCL_v1_slides.pdf](MPCL_v1_slides.pdf)

MPCL posits that latent representations acquire meaning by acting on the
outside world.

For continual learning to be manageable in complex environments and avoid
[catastrophic forgetting](https://github.com/rom1mouret/forgetful-networks),
meaning must remain stable over time. This is the core idea behind MPCL.

### Situated meaning

As the inputs and outputs of algorithms have no intrinsic meaning,
it is often the prerogative of the programmer to attach meaning to variables.

There are two kinds of meaning at play here.

1. meaning that emerges from the interplay with the environment. For instance,
frogs might view insects as mere calorie dispensers. Needless to say,
humans don't see insects the same way.
2. meaning from the programmer's perspective, which is roughly the same for
[everyone](https://en.wikipedia.org/wiki/Intersubjectivity).

Since the programmer's perspective (e.g. labels) is a byproduct of her environment,
it is not too much of a leap to treat her perspective as a proxy for her environment.
This is how I want to get away with explicitly modeling the environment in MPCL v1.

Take for instance a model categorizing x-ray images of tumors into malignant or benign.
If the model is deployed in a hospital, those labels have a tangible impact on the
environment. Labels bridge the gap between models and their deployment environment.
If you swap "malignant" with "benign" without changing the model's computation,
you end up with a very different situation.

So MPCL has two jobs:

- making sure the two kinds of meaning align. As we provide more and more training examples,
I expect the first kind of meaning to converge towards the second kind.
- making sure meaning remains stable over time.

In what follows, I will use "latent units" and "abstraction layer" interchangeably.
It refers to the last NN layer of the processing modules. 

### Continual learning (example)

Your system's training journey might start like that:
- module A: time T0 to T1: training the model to recognize human faces.
- module B: time T2 to T3: training the model to tell cats and dogs apart.
- module A: time T4 to T5: training the model to get better at recognizing human
faces, maybe in a novel context.

At time T1, your model and its abstraction layer are perfectly fit for recognizing
faces within domains it was trained on.

At time T2, your model moves on to learning to distinguish between cats and dogs.
It can build on module A's abstraction layer but it should not interfere with it
because the function of module A's abstraction layer is to recognize human faces, not
to recognize human faces AND pets. In programming terms, module A's network is
entirely frozen between T2 and T3.

At T4, we are back to face recognition.
Module A's internals can be safely refined so long as it keeps on delivering
abstractions that fulfill the same invariant function.
This might interfere with module B, but not in a
[destructive manner](https://en.wikipedia.org/wiki/Catastrophic_interference).

### Theoretical framework

In broad terms, we anchor latent representations by forcing them
to remain good at realizing the function they were originally trained for.
How the system realizes the
function-that-latent-representations-were-originally-trained-for *is* what
defines latent representations.

(By "function" I do not mean the mapping from sensory inputs to labels.
I mean the functions that map the abstraction layer to labels, or other kinds of outputs.
Recognizing faces in photographs fulfills the same function as recognizing faces
in real life.
Your friend's face is an abstraction in virtue of the realizability of the same
recognition function across many domains/contexts.)

Now, my goal is to formalize this idea and characterize the "meaning of latent
units" in more rigorous terms.

[link to MPCL Framework v1 pdf](MPCL-Framework-v1.pdf)


### Domain generalization

Both continual learning and [domain generalization](https://arxiv.org/pdf/2103.02503.pdf)
techniques try to create rich representations in a non-i.i.d. setting.
Domain generalization can be seen as a special case of
[domain-IL](https://arxiv.org/pdf/1904.07734v1.pdf) wherein most of
the hard work is done *before* observing out-of-distribution data.

One of the most promising approaches to domain generalization is causal
representation learning. The idea of uncovering the causal structure of the
problem is similar to MPCL's meaning alignment.

MPCL is akin to inverse domain generalization.
Instead of building high-quality abstraction hypotheses that are meant to
generalize well to unknown domains from the get go, the MPCL way would be to
try out hypotheses over many domains to assess their generalization power.
If they pass the quality test, they are entrusted with stronger, longer-lasting
connectivity with other modules.

# Proto-MPCL

Proto-MPCL is a solution to catastrophic forgetting wherein each input domain
is routed to its own dedicated processor. We rely on finding inconsistencies and
discrepancies to detect domain boundaries.

MPCL is not strongly committed to that way of overcoming forgetting,
but this multi-processor approach makes it easier to make sense of generalization,
abstraction and transferability from one domain to another.

I will attempt to frame bare-bones [Domain-IL and Class-IL](https://arxiv.org/pdf/1904.07734v1.pdf)
within this framework on MNIST and EMNIST datasets.
Admittedly, it takes a bit of shoehorning because simple systems don't create
many opportunities for Proto-MPCL to take advantage of the complexity to find
inconsistencies, for example by cross-checking predictions from multiple modules
of the system.  

Moreover, complex systems are generally upheld by high-dimensional latent
representations, with a [lot of empty space](https://en.wikipedia.org/wiki/Curse_of_dimensionality#Blessing_of_dimensionality)
wherein you would typically spot inconsistent configurations of latent values.

You will notice that the implemented system doesn't showcase any of the
inter-module [MPCL rules](MPCL_v1_slides.pdf).
This is because MNIST and EMNIST are not readily modularizable problems, so
we are essentially stuck with the one-module scenario.

Since we are not dealing with how modules are connected with one another,
I am dubbing the one-module case "Proto-MPCL". For MPCL to work at a larger scale,
we need proto-MPCL to do quite well on isolated continual-learning tasks. On paper,
it doesn't need to be perfect though. Like I said above, the more
modules you combine to satisfy various goals, the easier it gets to find
ways of detecting inconsistencies between the modules' outputs.

## Domain-IL classification on Permuted MNIST

In this scenario, the most straight-forward approach is to tether the
abstraction layer directly to the output classes.
To do so, we train a classifier to map the abstraction layer to classes on an arbitrary
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

Proto-MPCL doesn't lend itself well to Class-IL, but with a bit of trickery,
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

As with Domain-IL, the model was trained with `increment=1`, the most challenging scenario.

code: [emnist_class_il.py](emnist_class_il.py), [plot_emnist.py](plot_emnist.py)

The green curve shows how a collection of one-class classifiers would perform,
under some simplifying assumptions.

- false positive rate is 5%, in the same ballpark as my classifier.
- false negative rate is 0%.
- if multiple classifiers report a positive match, we randomly choose between them.

In simple settings such as MNIST and EMNIST, MPCL boils down to self-supervised
outlier detection. I believe it will prove more fruitful in more complex settings.

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
encounters something it doesn't know.  

I prefer `Triangle` because it doesn't throw anything away, not even noise.
It merely folds the input.

Disclaimer: I haven't studied in depth the dynamics of Triangle-activated
networks so it's possible that Triangle doesn't do what I think it does.
I have noticed improvements on Domain-IL though.

## Softmax

Softmaxing the last layer of the classifier was a mistake and I will try without
softmax in subsequent experiments.
Softmax is not a good choice for MPCL because there are infinite solutions to
`softmax(x) = y`, thus it doesn't constrain `x` enough for `x` (or anything
upstream of `x`) to be transferable to other modules.

It doesn't affect Permuted MNIST and EMNIST results that much because latent
representations are not transferred to other modules in our experimental setup.


## Terminology

A *group* of latent units has a *function*.
Within this group, individual latent units have a *meaning*.

It doesn't have to be called "meaning", but the word "meaning" is intuitive
for a number of reasons. First, it has an obvious connection with
intelligibility, a prerequisite to qualifying behavior patterns as rational
or intelligent.
Furthermore, the typical way of detecting inconsistencies is to look for configurations
of latent values that are *meaningless*, in the sense that they do not realize
any function.

I plan on extending MPCL to function-free intrinsic meaning (latent units that get
their meaning from adjacent units), in contrast to extrinsic meaning (latent
units that get their meaning from external labels/feedback).

## FAQ

##### > Must domain labels be known at training time?

Yes, this is how I have evaluated my models.
It is not a hard constraint from the framework though.

##### > Must domain labels be known at runtime?

No.

##### > Must module labels be known at runtime?

Yes. It is not a hard constraint from the framework either.

##### > Must task labels be known at runtime?

I find the idea of task confusing so it is no longer part of the framework.

There is a single task in Permuted MNIST, that of classifying digits.

The so-called Permuted MNIST tasks are treated as domains/contexts by MPCL,
so they don't need to be known at runtime.

##### > Can it be used for regression?

I haven't tried regression yet.
It comes with a few challenges.
1. If the regression model has only one numerical output,
it won't be enough to constrain multi-dimensional latent layers,
unless latent values are sparse or binary.
2. I am not sure what would be the best way to detect inconsistencies from numerical outputs.
Perhaps an ensemble of regressors could reveal discrepancies, or a Bayesian NN.

A workaround is to train the processors with a surrogate classification loss and have the classifier predict
both labels and the desired numerical targets at the same time.

##### > Can an MPCL system run at fixed capacity?

No, the system keeps growing as it learns new domains,
but infrequently used domains can be safely removed to free up some space.
Alternatively, they can be distilled down to smaller models.

##### > Can models be revisited later on when new training examples are made available?

Yes.


##### > If domains A and B are similar, does learning A help with learning B?

Not in vanilla MPCL.
But nothing stops you from implementing multi-task learning techniques orthogonally to MPLC.
For instance, you can [implement soft parameter sharing](https://ruder.io/multi-task/index.html#softparametersharing) between processors.

##### > Can trained abstraction layers be safely connected to other modules without interference risks?

Yes, that's the whole point of MPCL.
When new domains are learned, it is beneficial to downstream modules,
rather than destructive, as in this [zero-shot learning experiment](https://github.com/rom1mouret/domain_IL).

##### > Is there any limitation to the expressive power of MPCL models?

Feature processors can be arbitrarily complex,
though it is better if they don't aggressively filter noise out.
They don't have to be differentiable.
However, the [models mapping latent representations to external labels/actions are highly constrained](MPCL-Framework-v1.pdf).
In practice, linear mapping should do fine.

##### > Does MPCL operate on the same level as common continual learning algorithms such as [EWS](https://arxiv.org/pdf/1612.00796.pdf)?

Not exactly.
The starting point of MPCL is a principle as abstract as [Hebb's rule](https://en.wikipedia.org/wiki/Hebbian_theory) or the [free energy principle](https://en.wikipedia.org/wiki/Free_energy_principle).
Simply put, this principle states that situated meaning must remain stable across time.
MPCL is an attempt to derive an actionable framework from this (somewhat vague) principle.

Moreover, it is dealing with modules.  

##### > Must processors be trained one class at a time in the Class-IL setting?

Yes. If you get your data with `increment>1`, split the data into 1-increments.

##### > Isn't just glorified [one-class classification](https://en.wikipedia.org/wiki/One-class_classification) with a surrogate loss?

Maybe.
I haven't pushed MPCL far enough to see if it can bring something truly new to ML.
It would certainly look less like one-class classification if you were to apply it to highly modular architectures.

##### > Does AI have to be modular? Why not a single neural network?

I am agnostic to this question, but MPCL does rely on the assumption that
the system can be broken down into modules.

##### > Is MPCL biologically plausible?

Hopefully it is at some abstract level, but it is definitely not in any general sense
of biological plausibility.
For one thing, brains cannot allocate new feature processors out of thin air.
Also, the outside world is not labeled.

##### > How to train non-differentiable models within this framework?

Processors (inputs -> abstraction) and classifiers/regressors (abstraction -> targets) are typically trained conjointly.
This is where gradient descent shines, provided all the models involved are differentiable.

You may be able to get good results
with [coordinate descent](https://en.wikipedia.org/wiki/Coordinate_descent) on non-differentiable models.

Alternatively, if the classifiers/regressors are designed to be analytically invertible,
then you can calculate the latent values that correspond to the targets, and train processors to
predict the latent values as if they were the ground-truth.


##### > Isn't stabilizing meaning the same as stabilizing representations?

I mean "representation" in the ML sense, as in "[representation learning](https://en.wikipedia.org/wiki/Feature_learning)".
I do not mean "mental representation".

In that sense, if Representation-Preserving Continual Learning (RPCL) were a thing,
it would not be the same thing as MPCL. It would be more limited and more limiting.

Not every representation vector needs stability, whereas meaning always needs stability.
Also, representation vectors can be more fine-grained than meaning.

In a classification setting, each class' representation vectors need stability.
Meaning is not a particularly useful concept in such a setting because it is
conceivable to enumerate all the representation vectors that need stability
without resorting to any other concept.

In regression and motor settings, however, it becomes harder to identify the representations
that need stability`*` and it is useful to look at the problem from a meaning angle,
i.e. the link between representations and goals.
Goal-realizing functions need to remain *accurate*.
They need not remain *stable* at all times.

`*` I’m still debating whether it would be practical or not to require stability
for every single representation vector of the training set,
thereby doing away with meaning.
