from .collapsible_widget.collapsible_widget import CollapsibleWidget
from .gui_element import ButtonMenuAction, RenderedObjectSliders
from .window import NeuronPlotWindow, EngineWindow, LocationGroupInfoWindow
from PyQt6.QtWidgets import QApplication
import qdarktheme
from vispy.app import Application, Timer

from network import SpikingNeuronNetwork


class BaseApp(Application):

    def __init__(self, network: SpikingNeuronNetwork):

        native_app = QApplication([''])

        super().__init__(backend_name='pyqt6')

        self.network: SpikingNeuronNetwork = network

        self._plotting_config = self.network.plotting_config
        self._group_info_view_mode = self._plotting_config.group_info_view_mode

        native_app.setStyleSheet(qdarktheme.load_stylesheet())

        self.main_window: EngineWindow = self._init_main_window()

        # keep order for vbo numbers (2/3)

        if self._plotting_config.windowed_neuron_plots is True:
            self.neuron_plot_window = self._init_neuron_plot_window(network)
        else:
            self.neuron_plot_window = None

        if self._group_info_view_mode.windowed is True:
            self.location_group_info_window = self._init_windowed_group_info(network)
            self.group_info_panel = self.location_group_info_window.ui_panel_left
        else:
            self.location_group_info_window = None
            self.group_info_panel = self.main_window.ui_right

        self.time_elapsed_until_last_off = 0
        self.update_switch = False
        self.started = False
        self.timer_on = Timer('auto', connect=self.update, start=False)

        # noinspection PyUnresolvedReferences
        native_app.aboutToQuit.connect(self.network.unregister_registered_buffers)

    def set_main_context_as_current(self):
        self.main_window.scene_3d.set_current()

    def set_group_info_context_as_current(self):
        if self._group_info_view_mode.windowed is True:
            self.location_group_info_window.scene_3d.set_current()
        elif self._group_info_view_mode.split is True:
            self.main_window.group_info_scene.set_current()
        else:
            pass

    def _init_main_window(self) -> EngineWindow:
        main_window = EngineWindow(name="SNN Engine", app=self, plotting_config=self._plotting_config)

        main_window.scene_3d.set_current()
        for o in self.network.rendered_3d_objs:
            main_window.scene_3d.network_view.add(o)

        if main_window.scene_3d.voltage_plot:
            main_window.scene_3d.voltage_plot.add(self.network.voltage_plot)
        if main_window.scene_3d.scatter_plot:
            main_window.scene_3d.scatter_plot.add(self.network.firing_scatter_plot)

        if self._group_info_view_mode.scene is True:
            main_window.scene_3d.group_firings_plot.add(self.network.group_firing_counts_plot)

        if main_window.scene_3d.group_firings_plot_single0 is not None:
            main_window.scene_3d.group_firings_plot_single0.add(self.network.group_firing_counts_plot_single0)
        if main_window.scene_3d.group_firings_plot_single1 is not None:
            main_window.scene_3d.group_firings_plot_single1.add(self.network.group_firing_counts_plot_single1)

        if self._group_info_view_mode.split is True:
            # keep order for vbo id
            main_window.add_group_info_scene_to_splitter(self._plotting_config)
            # keep order for vbo id
            main_window.group_info_scene.group_firings_plot.add(self.network.group_firing_counts_plot)
            main_window.group_info_scene.view.add(self.network.group_info_mesh)

        main_window.set_keys({
            'left': self.network.selector_box.translate.mv_left,
            'right': self.network.selector_box.translate.mv_right,
            'up': self.network.selector_box.translate.mv_fw,
            'down': self.network.selector_box.translate.mv_bw,
            'pageup': self.network.selector_box.translate.mv_up,
            'pagedown': self.network.selector_box.translate.mv_down,
        })
        main_window.show()
        return main_window

    def _init_neuron_plot_window(self, network: SpikingNeuronNetwork):
        neuron_plot_window = NeuronPlotWindow(name='Neuron Plots', app=self, plotting_config=self._plotting_config)
        neuron_plot_window.voltage_plot_sc.set_current()
        neuron_plot_window.voltage_plot_sc.plot.view.add(network.voltage_plot)
        neuron_plot_window.scatter_plot_sc.set_current()
        neuron_plot_window.scatter_plot_sc.plot.view.add(network.firing_scatter_plot)
        neuron_plot_window.show()
        return neuron_plot_window

    def _init_windowed_group_info(self, network: SpikingNeuronNetwork):
        location_group_info_window = LocationGroupInfoWindow(
            "Location Groups", app=self, parent=self.main_window, plotting_config=self._plotting_config)
        location_group_info_window.scene_3d.set_current()
        location_group_info_window.scene_3d.view.add(network.group_info_mesh)

        if location_group_info_window.scene_3d.group_firings_plot is not None:
            location_group_info_window.scene_3d.group_firings_plot.add(self.network.group_firing_counts_plot)

        location_group_info_window.show()
        return location_group_info_window

    def _bind_ui(self):
        network_config = self.network.network_config

        self._connect_main_buttons_and_actions()

        self.main_ui_panel.add_3d_object_sliders(self.network.selector_box)
        self.main_ui_panel.add_3d_object_sliders(self.network.input_cells)
        self.main_ui_panel.add_3d_object_sliders(self.network.output_cells)

        self._connect_g_props_sliders(network_config)

        self.main_ui_panel.sliders.sensory_weight.connect_property(
            self.network.input_cells,
            self.network.input_cells.src_weight)

        self.group_info_panel.group_ids_combobox().add_items(self.network.group_info_mesh.group_id_texts.keys())
        self.group_info_panel.group_ids_combobox.connect(self.group_id_combo_box_text_changed)

        self.group_info_panel.g_flags_combobox().add_items(self.network.group_info_mesh.G_flags_texts.keys())
        self.group_info_panel.g_flags_combobox.connect(self.g_flags_combo_box_text_changed)

        self.group_info_panel.g_props_combobox().add_items(self.network.group_info_mesh.G_props_texts.keys())
        self.group_info_panel.g_props_combobox.connect(self.g_props_combo_box_text_changed)

        self.group_info_panel.combo_boxes_collapsible.toggle_collapsed()

    def _connect_main_buttons_and_actions(self):
        self.main_ui_panel.buttons.start.clicked.connect(self.trigger_update_switch)
        self.main_ui_panel.buttons.pause.clicked.connect(self.trigger_update_switch)
        self.main_ui_panel.buttons.exit.clicked.connect(self.quit)
        self.main_ui_panel.buttons.toggle_outergrid.clicked.connect(self.toggle_outergrid)
        self.actions.start.triggered.connect(self.trigger_update_switch)
        self.actions.pause.triggered.connect(self.trigger_update_switch)
        self.actions.exit.triggered.connect(self.quit)
        self.actions.toggle_outergrid.triggered.connect(self.toggle_outergrid)

    def _connect_g_props_sliders(self, network_config):
        self.main_ui_panel.sliders.thalamic_inh_input_current.connect_property(
            self.network.GPU.G_props,
            network_config.InitValues.ThalamicInput.inh_current)
        self.main_ui_panel.sliders.thalamic_exc_input_current.connect_property(
            self.network.GPU.G_props,
            network_config.InitValues.ThalamicInput.exc_current)
        self.main_ui_panel.sliders.sensory_input_current0.connect_property(
            self.network.GPU.G_props,
            network_config.InitValues.SensoryInput.input_current0)
        self.main_ui_panel.sliders.sensory_input_current1.connect_property(
            self.network.GPU.G_props,
            network_config.InitValues.SensoryInput.input_current1)

    def group_id_combo_box_text_changed(self, s):
        print(s)
        self.network.group_info_mesh.set_group_id_text(s)

    def g_flags_combo_box_text_changed(self, s=None):
        if s is None:
            s = self.group_info_panel.g_flags_combobox().currentText()
        print(s)
        self.network.group_info_mesh.set_g_flags_text(s)

    def g_props_combo_box_text_changed(self, s):
        print(s)
        if not isinstance(s, str):
            s = self.group_info_panel.g_props_combobox().currentText()
        self.network.group_info_mesh.set_g_props_text(s)

    def toggle_outergrid(self):
        self.network.outer_grid.visible = not self.network.outer_grid.visible
        if self.network.outer_grid.visible is True:
            # self.buttons.toggle_outergrid.setText('Hide OuterGrid')
            self.main_ui_panel.buttons.toggle_outergrid.setChecked(True)
            self.actions.toggle_outergrid.setChecked(True)

            if self._group_info_view_mode.scene is True:
                self.main_window.scene_3d.group_firings_plot.visible = True
            if self.neuron_plot_window is not None:
                self.neuron_plot_window.hide()

            self.network.group_info_mesh.group_id_text_visual.pos += 1

        else:
            # self.buttons.toggle_outergrid.setText('Show OuterGrid')
            self.main_ui_panel.buttons.toggle_outergrid.setChecked(False)
            self.actions.toggle_outergrid.setChecked(False)
            if self._group_info_view_mode.scene is True:
                self.main_window.scene_3d.group_firings_plot.visible = False
            if self.neuron_plot_window is not None:
                self.neuron_plot_window.show()

    def trigger_update_switch(self):
        self.update_switch = not self.update_switch
        if self.update_switch is True:
            self.timer_on.start()
            self.main_ui_panel.buttons.start.setDisabled(True)
            self.actions.start.setDisabled(True)
            self.main_ui_panel.buttons.pause.setDisabled(False)
            self.actions.pause.setDisabled(False)
        else:
            self.time_elapsed_until_last_off += self.timer_on.elapsed
            self.timer_on.stop()
            self.main_ui_panel.buttons.start.setDisabled(False)
            self.actions.start.setDisabled(False)
            self.main_ui_panel.buttons.pause.setDisabled(True)
            self.actions.pause.setDisabled(True)

    @property
    def main_ui_panel(self):
        return self.main_window.ui_panel_left

    @property
    def actions(self):
        return self.main_window.menubar.actions

    # noinspection PyUnusedLocal
    def update(self, event):
        if self.update_switch is True:
            self.network.GPU.update()
            t = str(self.network.GPU.Simulation.t)
            if self.neuron_plot_window is not None:
                self.neuron_plot_window.voltage_plot_sc.table.t.text = t
                self.neuron_plot_window.scatter_plot_sc.table.t.text = t
            if self._group_info_view_mode.split is True:
                self.main_window.group_info_scene.table.t.text = t
            self.main_window.scene_3d.table.t.text = t
            self.main_window.scene_3d.table.update_duration.text = str(self.network.GPU.Simulation.update_duration)
