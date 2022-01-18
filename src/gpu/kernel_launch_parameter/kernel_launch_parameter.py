from pycuda import driver


class DeviceProperties:

    def __init__(self, device):

        self.device = driver.Device(device)

        self.warp_size = self.device.get_attribute(driver.device_attribute.WARP_SIZE)
        self.max_ts_per_block = self.device.get_attribute(driver.device_attribute.MAX_THREADS_PER_BLOCK)
        self.max_ts_per_mp = self.device.get_attribute(driver.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR)

        # self.max_block_dim_x = dev.get_attribute(driver.device_attribute.MAX_BLOCK_DIM_X)
        # self.max_block_dim_y = dev.get_attribute(driver.device_attribute.MAX_BLOCK_DIM_Y)
        # self.max_block_dim_z = dev.get_attribute(driver.device_attribute.MAX_BLOCK_DIM_Z)

