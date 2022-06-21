from network.network_config import NetworkConfig, PlottingConfig
from network.network_states import LocationGroupProperties


class NetworkArrayShapes:

    # noinspection PyPep8Naming
    def __init__(self,
                 config: NetworkConfig,
                 T: int,
                 n_N_states: int,
                 plotting_config: PlottingConfig,
                 n_neuron_types=2):

        self.N_pos = (config.N, config.vispy_scatter_plot_stride)
        self.N_G = (config.N, config.N_G_n_cols)

        # Network Representation
        self.N_rep = (config.S, config.N)  # float
        self.N_weights = self.N_rep  # np.float32
        self.N_delays = (config.D + 1, config.N)  # int
        self.N_fired = (1, config.N)  # int
        self.firing_times = (15, config.N)  # float
        self.firing_idcs = self.firing_times  # dtype=np.int32
        self.firing_counts = 2 * T  # dtype=np.int32

        # pt, u, v, a, b, c, d, I
        self.N_states = (n_N_states, config.N)  # dtype=np.float32

        # GROUPS (location-based)

        self.G_pos = (config.G, 3)  # position of each location group; dtype=np.int32
        self.G_rep = (config.G, config.G)
        self.G_delay_counts = (config.G, config.D + 1)  # number of groups per delays; dtype=np.int32
        # G_neuron_counts-layout:
        #   - columns: location-based group-ids
        #   - row 0 to max(#neuron-types) - 1: #neurons in this group and type
        #   - row max(#neuron-types) to last row:
        #   for x in neuron-types:
        #       row0 = max(#neuron-types) + (D * (neuron-type) - 1)
        #       row1 = row0 + D
        #       rows from row0 to row1:
        #           #neurons for this group per delay of type 1
        #
        # Example:
        #
        # max(delay) = D - 1
        # #(neurons) = 20
        # #(neurons of neuron-type 1) = 4
        # #(neurons of neuron-type 2) = 16
        # n_neuron_types = 2
        #
        # tensor([[ 0,  0,  0,  2,  0,  0,  1,  1],  # sum = 4
        #         [ 4,  3,  1,  1,  3,  0,  3,  1],  # sum = 16
        #         [ 0,  0,  0,  2,  0,  0,  1,  1],  # row0-0
        #         [ 4,  4,  4,  2,  4,  4,  3,  3],  # row1-0
        #         [ 4,  3,  1,  1,  3,  0,  3,  1],  # row0-1
        #         [12, 13, 15, 15, 13, 16, 13, 15]]) # row1-1
        #
        # (group ids are 0-indexed)
        # (neuron type ids are 1-indexed)
        # #(neurons with (delay-from-group4==1)) = G_delay_counts[1:4]
        # #(neurons-type-x with (delay-from-group_y==z)) =
        #   G_delay_counts[n_neuron_types + (D * (x-1)) + z  :y]

        self.G_neuron_counts = (n_neuron_types + n_neuron_types * config.D, config.G)  # dtype=np.int32
        # self.G_neuron_typed_ccount = (1, 2 * G + 1)  # dtype=np.int32

        syn_count_shape = (n_neuron_types * (config.D + 1), config.G)
        # expected (cumulative) count of synapses per source types and delay (sources types: inhibitory or excitatory)
        self.G_exp_ccsyn_per_src_type_and_delay = syn_count_shape  # dtype=np.int32

        # expected cumulative sum of excitatory synapses per delay and per sink type
        # (sink types: inhibitory, excitatory)
        self.G_exp_exc_ccsyn_per_snk_type_and_delay = syn_count_shape  # dtype=np.int32

        self.G_conn_probs = (n_neuron_types * config.G, config.D)  # dtype=np.float32
        # self.relative_autapse_idcs = (3 * D, G)  # dtype=np.int32

        # dtype=np.int32;  selected_p, thalamic input (on/off), ...
        self.G_props = (LocationGroupProperties.__len__(), config.G)

        self.voltage_plot = (plotting_config.n_voltage_plots * plotting_config.voltage_plot_length, 2)

        self.n_group_separator_lines = 2 * config.G + 1
        self.plot_group_line_pos = (2 * self.n_group_separator_lines, 2)
        self.plot_group_line_colors = (2 * self.n_group_separator_lines, 4)

        self.firings_scatter_plot = (plotting_config.n_scatter_plots * plotting_config.scatter_plot_length,
                                     config.vispy_scatter_plot_stride)

        self.voltage_plot_map = plotting_config.n_voltage_plots
        self.firings_scatter_plot_map = plotting_config.n_scatter_plots

        assert config.G == self.G_props[1]
        assert config.N == self.N_states[1]

    # noinspection PyPep8Naming
    @property
    def N_rep_inv(self):
        return self.N_rep[1], self.N_rep[0]
