import pandas as pd
import scipy.io

class Bus:
    def __init__(self, id, name, region, load_factor, load):
        self.id = id
        self.name = name
        self.region = region
        self.load_factor = load_factor
        self.load = load

    def calculate_load(self, day, hour, load_data_mat):
        # load_data_mat = scipy.io.loadmat('data_PROSPECT43/load_PROSPECT43.mat')
        region = f'load_{self.region}'
        self.load = self.load_factor * load_data_mat[region][0,day,hour,5]
        return self.load


class Line:
    def __init__(self, id, name, from_bus_id, to_bus_id, R, X, halfB, voltage, capacity, length):
        self.id = id
        self.name = name
        self.from_bus_id = from_bus_id
        self.to_bus_id = to_bus_id
        self.R = R
        self.X = X
        self.halfB = halfB
        self.voltage = voltage
        self.capacity = capacity
        self.length = length


class Gen:
    def __init__(self, id, bus_id, type, region, capacity, upper, min_output_rate, ramp, max_start_time, max_shut_time, fixcost, outcost, startcost, last_output):
        self.id = id
        self.bus_id = bus_id
        self.type = type
        self.region = region
        self.capacity = capacity
        self.upper = upper
        self.min_output_rate = min_output_rate
        self.ramp = ramp
        self.max_start_time = max_start_time
        self.max_shut_time = max_shut_time
        self.fixcost = fixcost
        self.outcost = outcost
        self.startcost = startcost
        self.last_output = last_output
    
    def calculate_upper(self, day, hour, vre_data_mat):
        if self.type == 11 or self.type == 12 or self.type == 14 or self.type == 15:
            # vre_data_mat = scipy.io.loadmat('data_PROSPECT43/vre_PROSPECT43.mat')
            if self.type == 11:
                case = 'solar_case'
                region = f'solar_{self.region}'
            elif self.type == 12:
                case = 'solar_dis_case'
                region = f'solar_dis_{self.region}'
            elif self.type == 14:
                case = 'wind_case'
                region = f'wind_{self.region}'
            elif self.type == 15:
                case = 'wind_ofs_case'
                region = f'wind_ofs_{self.region}'
            self.upper = self.capacity * vre_data_mat[case][0,0][region][0,day,hour] + 0.01
            # 不是哥们我真的这辈子想不出+ 0.01这招
            # 真草了我先不管为什么了，暂时这样吧
        else:
            self.upper = self.capacity
        return self.upper

    def record_last_output(self, last_output):
        self.last_output = last_output

class Storage:
    def __init__(self, id, bus_id, capacity, efficiency, storage_time, soc):
        self.id = id
        self.bus_id = bus_id
        self.capacity = capacity
        self.efficiency = efficiency
        self.storage_time = storage_time
        self.soc = soc