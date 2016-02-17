.. _implement_algo:

===========================
Implementing New Algorithms
===========================

In this section, we will walk through the implementation of the classical
REINFORCE [1]_ algorithm, also known as the "vanilla" policy gradient.
We will exploit the utilities provided by the framework whenever possible.

Preliminaries
=============

First, let's briefly review the algorithm along with some notations. We work
with an MDP defined by the tuple :math:`(\mathcal{S}, \mathcal{A}, P, r, \mu_0, \gamma, T)`, where
:math:`\mathcal{S}` is a set of states, :math:`\mathcal{A}` is a set of
actions, :math:`P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]`
is the transition probability, :math:`r: \mathcal{S} \times \mathcal{A}
\to \mathbb{R}` is the reward function, :math:`\mu_0: \mathcal{S} \to [0, 1]`
is the initial state distribution, :math:`\gamma \in [0, 1]` is the discount
factor, and :math:`T \in \mathbb{N}` is the horizon. REINFORCE directly
optimizes a parameterized stochastic policy
:math:`\pi_\theta: \mathcal{S} \times \mathcal{A} \to [0, 1]` by performing
gradient ascent on the expected return objective:

.. math::
    
    \eta(\theta) = \mathbb{E}\left[\sum_{t=0}^T \gamma^t r(s_t, a_t)\right]

where the expectation is implicitly taken over all possible trajectories,
following the sampling procedure :math:`s_0 \sim \mu_0`,
:math:`a_t \sim \pi_\theta(\cdot | s_t)`, and
:math:`s_{t+1} \sim P(\cdot | s_t, a_t)`. By the likelihood ratio trick,
the gradient of the objective with respect to :math:`\theta` is given by

.. math::
    
    \nabla_\theta \eta(\theta) = \mathbb{E}\left[\left(\sum_{t=0}^T \gamma^t r(s_t, a_t)\right) \left(\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \right)\right]

We can reduce the variance of this estimator by noting that for :math:`t' < t`,

.. math::

    \mathbb{E}\left[ r(s_{t'}, a_{t'}) \nabla_\theta \log \pi_\theta(a_t | s_t) \right] = 0

Hence,

.. math::
    
    \nabla_\theta \eta(\theta) = \mathbb{E}\left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \sum_{t'=t}^T \gamma^{t'} r(s_{t'}, a_{t'}) \right]

Often, we use the following estimator instead:

.. math::
    
    \nabla_\theta \eta(\theta) = \mathbb{E}\left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \sum_{t'=t}^T \gamma^{t'-t} r(s_{t'}, a_{t'}) \right]

where :math:`\gamma^{t'}` is replaced by :math:`\gamma^{t'-t}`. When viewing the discount factor as a variance reduction factor for the undiscounted objective, this alternative gradient estimator has less bias, at the expense of having a larger variance.

We can further reduce the variance by subtracting a baseline :math:`b(s_t)`
from the empirical return :math:`\sum_{t'=t}^T \gamma^{t'-t} r(s_{t'}, a_{t'})`:

.. math::
    
    \nabla_\theta \eta(\theta) = \mathbb{E}\left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \left(\sum_{t'=t}^T \gamma^{t'-t} r(s_{t'}, a_{t'}) - b(s_{t}) \right) \right]

The baseline :math:`b(s_t)` is typically implemented as an estimator of
:math:`V^\pi(s_t)`. The above formula will be the central object of our
implementation.


Constructing the Computation Graph
==================================


.. [1] Williams, Ronald J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
