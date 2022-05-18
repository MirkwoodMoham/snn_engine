import numba.cuda
from vispy import gloo
from vispy.app import Timer
import sys

from network import SpikingNeuronNetwork, NetworkConfig, PlottingConfig
from gui import (
    EngineWindow,
    BackendApp,
    RenderedObjectSliders
)
# from simulation import vbodata2host
from pycuda import autoinit


class EngineConfig:

    N: int = 10 * 10 ** 3
    T: int = 2000  # Max simulation duration

    device: int = 0

    max_batch_size_mb: int = 3000

    network_config = NetworkConfig(N=N, N_pos_shape=(4, 4, 1))
    plotting_config = PlottingConfig(N=N,
                                     n_voltage_plots=10, voltage_plot_length=100,
                                     n_scatter_plots=1000, scatter_plot_length=1000)


class Engine:

    def __init__(self):

        numba.cuda.select_device(EngineConfig.device)

        self.app = BackendApp()

        # keep order for vbo numbers (1/3)
        self.network = SpikingNeuronNetwork(
            network_config=EngineConfig.network_config,
            plotting_config=EngineConfig.plotting_config,
            max_batch_size_mb=EngineConfig.max_batch_size_mb,
            T=EngineConfig.T)

        # keep order for vbo numbers (2/3)
        self.window = EngineWindow(name="SNN Engine",
                                   app=self.app.vs,
                                   network=self.network)

        # keep order for vbo numbers (3/3)
        self.network.initialize_GPU_arrays(EngineConfig.device)

        # self.window.scene_3d.add_to_network_view(self.network.selected_group_boxes.obj)

        self.window.set_keys({
            'left': self.network.selector_box.translate.mv_left,
            'right': self.network.selector_box.translate.mv_right,
            'up': self.network.selector_box.translate.mv_fw,
            'down': self.network.selector_box.translate.mv_bw,
            'pageup': self.network.selector_box.translate.mv_up,
            'pagedown': self.network.selector_box.translate.mv_down,
        })

        selector_box_collapsible = RenderedObjectSliders(self.network.selector_box, self.window)
        input_cell_collapsible = RenderedObjectSliders(self.network.input_cells, self.window)
        output_cell_collapsible = RenderedObjectSliders(self.network.output_cells, self.window)

        self.ui_left.objects_collapsible.add(selector_box_collapsible)
        self.ui_left.objects_collapsible.add(input_cell_collapsible)
        self.ui_left.objects_collapsible.add(output_cell_collapsible)

        self.ui_left.sensory_input_collapsible.toggle_collapsed()
        self.ui_left.objects_collapsible.toggle_collapsed()
        selector_box_collapsible.toggle_collapsed()
        # self.ui_left.thalamic_input_collapsible.toggle_collapsed()

        self.time_elapsed_until_last_off = 0
        # self.set_scale(0)
        self.update_switch = False
        self.started = False
        self.timer_on = Timer('auto', connect=self.update, start=False)

        self.buttons.start.clicked.connect(self.trigger_update_switch)
        self.buttons.pause.clicked.connect(self.trigger_update_switch)
        self.actions.start.triggered.connect(self.trigger_update_switch)
        self.actions.pause.triggered.connect(self.trigger_update_switch)

        self.buttons.toggle_outergrid.clicked.connect(self.toggle_outergrid)
        self.actions.toggle_outergrid.triggered.connect(self.toggle_outergrid)

        self.sliders.thalamic_inh_input_current.connect_property(
            self.network.GPU.G_props,
            EngineConfig.network_config.InitValues.ThalamicInput.inh_current)

        self.sliders.thalamic_exc_input_current.connect_property(
            self.network.GPU.G_props,
            EngineConfig.network_config.InitValues.ThalamicInput.exc_current)

        self.sliders.sensory_input_current0.connect_property(
            self.network.GPU.G_props,
            EngineConfig.network_config.InitValues.SensoryInput.input_current0)

        self.sliders.sensory_input_current1.connect_property(
            self.network.GPU.G_props,
            EngineConfig.network_config.InitValues.SensoryInput.input_current1)

        self.sliders.sensory_weight.connect_property(
            self.network.input_cells,
            self.network.input_cells.src_weight)

    @property
    def actions(self):
        return self.window.ui.menubar.actions

    @property
    def buttons(self):
        return self.ui_left.buttons

    @property
    def sliders(self):
        return self.ui_left.sliders

    @property
    def ui_left(self):
        return self.window.ui.ui_left

    def run(self):
        self.app.vs.run()

    def update(self, event):
        if self.update_switch is True:
            # elapsed = event.elapsed + self.time_elapsed_until_last_off
            # self.set_scale(elapsed)
            self.network.GPU.update()
            self.window.scene_3d.time_txt2.text = str(self.network.GPU.Simulation.t)

    def toggle_outergrid(self):
        self.network.outer_grid.visible = not self.network.outer_grid.visible
        if self.network.outer_grid.visible is True:
            # self.buttons.toggle_outergrid.setText('Hide OuterGrid')
            self.buttons.toggle_outergrid.setChecked(True)
            self.actions.toggle_outergrid.setChecked(True)
        else:
            # self.buttons.toggle_outergrid.setText('Show OuterGrid')
            self.buttons.toggle_outergrid.setChecked(False)
            self.actions.toggle_outergrid.setChecked(False)

        from vispy.visuals.transforms import STTransform, TransformSystem
        # a: STTransform = self.network.selected_group_boxes.obj.transform
        # a.move((0, 0, .5))
        print()

    def trigger_update_switch(self):
        self.update_switch = not self.update_switch
        if self.update_switch is True:
            self.timer_on.start()
            self.buttons.start.setDisabled(True)
            self.actions.start.setDisabled(True)
            self.buttons.pause.setDisabled(False)
            self.actions.pause.setDisabled(False)
            # self.buttons.start.setText('Pause')
        else:
            self.time_elapsed_until_last_off += self.timer_on.elapsed
            self.timer_on.stop()
            self.buttons.start.setDisabled(False)
            self.actions.start.setDisabled(False)
            self.buttons.pause.setDisabled(True)
            self.actions.pause.setDisabled(True)
            # self.buttons.start.setText('Start')


if __name__ == '__main__':

    gloo.gl.use_gl('gl+')

    eng = Engine()

    # gloo.set_state(cull_face=True, depth_test=True, blend=True)
    if sys.flags.interactive != 1:
        eng.run()
