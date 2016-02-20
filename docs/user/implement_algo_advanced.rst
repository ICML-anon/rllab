.. _implement_algo_advanced:

======================================
Implementing New Algorithms (Advanced)
======================================

In this section, we will anatomize the implementation of vanilla policy gradient
algorithm provided in the algorithm, available at :code:`rllab/algo/vpg.py`. It utilizes
many functionalities provided by the framework, which we describe below.


The :code:`BatchPolopt` Class
=======================

The :code:`VPG` class inherits from :code:`BatchPolopt`, which is an abstract
class inherited by algorithms with a common structure. The structure is as
follows:

- Initialize policy :math:`\pi` with parameter :math:`\theta_1`.

- Initialize the computational graph structure.

- For iteration :math:`k = 1, 2, \ldots`:

    - Sample N trajectories :math:`\tau_1`, ..., :math:`\tau_n` under the
      current policy :math:`\theta_k`, where
      :math:`\tau_i = (s_t^i, a_t^i, R_t^i)_{t=0}^{T-1}`. Note that the last
      state is dropped since no action is taken after observing the last state.

    - Update the policy based on the collected on-policy trajectories.

    - Print diagnostic information and store intermediate results.

Note the parallel between the structure above and the pseudocode for VPG. The
:code:`BatchPolopt` class takes care of collecting samples and common diagnostic
information. It also provides an abstraction of the general procedure above, so
that algorithm implementations only need to fill the missing pieces. The core
of the :code:`BatchPolopt` class is the :code:`train()` method:


.. code-block:: py

    def train(self, mdp, policy, baseline, **kwargs):
        # ...
        opt_info = self.init_opt(mdp, policy, baseline)
        for itr in xrange(self.start_itr, self.n_itr):
            samples_data = self.obtain_samples(itr, mdp, policy, baseline)
            opt_info = self.optimize_policy(itr, policy, samples_data, opt_info)
            params = self.get_itr_snapshot(
                itr, mdp, policy, baseline, samples_data, opt_info)
            logger.save_itr_params(itr, params)
            # ...

The :code:`obtain_samples` is implemented. The derived class needs to provide
implementation for :code:`init_opt`, which initializes the computation graph,
:code:`optimize_policy`, which updates the policy based on the collected data,
and :code:`get_itr_snapshot`, which returns a dictionary of objects to be persisted
per iteration.

The :code:`BatchPolopt` powers quite a few algorithms:

- Vanilla Policy Gradient: :code:`rllab/algo/vpg.py`

- Natural Policy Gradient: :code:`rllab/algo/npg.py`

- Reward-Weighted Regression: :code:`rllab/algo/erwr.py`

- Trust Region Policy Optimization: :code:`rllab/algo/trpo.py`

- Relative Entropy Policy Search: :code:`rllab/algo/reps.py`


Parallel Sampling
=================


Command-line Arguments
======================


Logging
=======
