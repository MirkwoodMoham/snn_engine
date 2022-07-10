========
Notation
========


Network Constants
=================

.. math::

    \begin{eqnarray}
        N &:& \text{number of Neurons}, \\
        G &:& \text{number of location-groups}, \\
        D &:& \text{maximal Delay}, \\
        S &:& \text{number of synapses per neuron}.
    \end{eqnarray}


Variable Naming
===============


.. math::

    \begin{eqnarray}
        n &:& \text{neuron(-ID)}; \quad n \in \{0, ..., N-1\}, \\
        g &:& \text{group(-ID)}; \quad g \in \{0, ..., G-1\} \\
        s &:& \text{synapse(-ID)}; \quad s \in \{0, ..., S-1\} \\
        d &:& \text{delay}; \quad d \in \{0, ..., D-1\} \\
        \tau &:& \text{neuron type}; \quad \tau \in \{1, 2\} \text{ or } \{inh, exc\}\\
        N_{pre\text{-}src} &:& \text{ ID of a pre-synaptic Neuron to a (source) neuron,} \\
        N_{src} &:& \text{ ID of a pre-synaptic (source) neuron,} \\
        N_{snk} &:& \text{ ID of a post-synaptic (sink) neuron,} \\
        G_{src} &:& \text{ group-ID of a pre-synaptic neuron,} \\
        G_{snk} &:& \text{ group-ID of a post-synaptic neuron}
        .\\
    \end{eqnarray}

Network-Arrays
==============


.. math::

    \begin{eqnarray}
        \text{NeuronCount}_D[d, g] &:&
        \text{Number of Neurons (potential synapses) per Delay w.r.t. a group }g,\\
        \text{TypedNeuronCount}_D[d, g, \tau] &:&
        \text{Number of Neurons (potential synapses) per Delay w.r.t. a group }g \text{ of type }\tau,\\
        \text{Rep}_{\text{}N, S}[n, s] &:& \text{ID of the post-synaptic neuron of the neuron }
        n \text{ at the synapse } s;\\
        & & \text{ Network representation } (\text{shape: } N \times S), \\
        \text{W}_{N,S}[n, s] &:& \text{weight of the synapse at } \text{Rep}_{\text{}N}(n, s)
        \text{ (shape: }N \times S), \\
        \text{Rep}_{\text{}pre}[i] &:& \text{indices of synapses } (\text{w.r.t. } \text{Rep}_{\text{}N})
        \text{ directed at a neuron } (\text{shape: }NS); \\
        \text{Idcs}_{pre, N}[n] &:& \text{first index for a neuron } n \text{ in. Rep}_{pre};\\
        & & \text{interval: }  \big[\text{Idcs}_{pre}[n], \text{Idcs}_{pre}[n+1] - 1\big]
        \quad (\text{flatten}(\text{Rep}_{\text{}N})[\text{Idcs}_{pre}[n]] = n )\\
        \text{Group}_N[n] &:& \text{ group-ID of a neuron } n, \\
        \text{Type}_N[n] &:& \text{type of a neuron } n \text{ (inhibitory (1) or excitatory (2)),}\\
        \text{LastFiringTime}_N &:& \text{ last known firing time of a Neuron,} \\
        \text{STDPConfig}_G &:& \text{Current group to group STDP-Configuration (-1, 0 or 1) }
            (\text{shape: }G \times G)
        .\\
    \end{eqnarray}
