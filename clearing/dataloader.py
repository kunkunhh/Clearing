"""
数据读取模块 - 从Excel文件读取数据并创建对象列表
其他代码可以用from dataloader import buses, lines方式从此文件读取
"""

import pandas as pd
from data import Bus, Line, Gen, Storage

# 全局变量，存储读取的数据
buses = []
lines = []
gens = []
storages = []

def load_data(file_path='data_PROSPECT43/PROSPECT43_data.xlsx'):
    """从Excel文件加载数据并创建对象列表"""
    global buses, lines, gens, storages
    
    try:
        # 获取所有工作表名称
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        
        # 读取所有工作表到字典
        all_sheets = {}
        for sheet in sheet_names:
            all_sheets[sheet] = pd.read_excel(file_path, sheet_name=sheet)
        
        # 获取各个数据表
        bus_df = all_sheets.get('Bus')
        lines_df = all_sheets.get('Lines')
        generation_units_df = all_sheets.get('Generation Units')
        storage_units_df = all_sheets.get('Storage Units')
        
        # 创建对象列表
        buses = create_bus_objects(bus_df)
        lines = create_line_objects(lines_df)
        gens = create_gen_objects(generation_units_df)
        storages = create_storage_objects(storage_units_df, all_sheets)
        
        # print(f"成功创建对象:")
        # print(f"- Bus对象: {len(buses)} 个")
        # print(f"- Line对象: {len(lines)} 个") 
        # print(f"- Gen对象: {len(gens)} 个")
        # print(f"- Storage对象: {len(storages)} 个")

        # 手动切片
        manual_slice_objects()
        
        return True
        
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")
        return False
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False

def manual_slice_objects():
    # 手动选取
    global buses, lines, gens, storages
    buses = buses[:40]
    lines = lines[:77]
    gens = gens[:214]
    storages = storages[:21]

def create_bus_objects(bus_df):
    """创建Bus对象列表"""
    buses = []
    
    for index, row in bus_df.iterrows():
        try:
            # 处理Bus数据
            bus_id = int(row['Bus No.'])
            bus_name = str(row['Name of Bus'])
            bus_region = str(row['Region'])
            load_factor = float(row['Bus Load Participation Factor \nby Region']) 
            bus_load = load_factor
            if 1<=bus_id<=4:
                bus_load *= 34.84 * 10**3
            elif 5<=bus_id<=17:
                bus_load *= 154.65 * 10**3
            elif 18<=bus_id<=28:
                bus_load *= 135.40 * 10**3
            elif 29<=bus_id<=36:
                bus_load *= 63.49 * 10**3
            else:
                bus_load *= 56.97 * 10**3
            
            bus = Bus(bus_id, bus_name, bus_region, load_factor, bus_load)
            buses.append(bus)
        except (ValueError, TypeError) as e:
            print(f"创建Bus对象时出错 (行 {index+1}): {e}")
            continue
    
    return buses

def create_line_objects(lines_df):
    """创建Line对象列表"""
    lines = []
    
    for index, row in lines_df.iterrows():
        try:
            # 处理Line数据
            line_id = int(row['Line No.'])
            line_name = str(row['Name of line'])
            from_bus_id = int(row['From bus No.'])
            to_bus_id = int(row['To bus No.'])
            R = float(row['Resistance R (p.u.)'])
            X = float(row['Reactance X (p.u.)'])
            halfB = float(row['Half of Susceptance B/2 (p.u.)'])
            voltage = float(row['Voltage (kV)'])
            capacity = float(row['Line capacity (MW)'])
            length = float(row['Length (km)'])
            
            line = Line(line_id, line_name, from_bus_id, to_bus_id, R, X, halfB, voltage, capacity, length)
            lines.append(line)
        except (ValueError, TypeError) as e:
            # print(f"创建Line对象 时出错 (行 {index+1}): {e}")
            continue
    
    return lines

def create_gen_objects(gen_df):
    """创建Gen对象列表"""
    gens = []
    
    for index, row in gen_df.iterrows():
        try:
            # 处理Gen数据
            gen_id = int(row['Unit No.'])
            bus_id = int(row['Bus No.'])
            type = int(row['Unit Type'])
            region = str(row['Region'])
            capacity = float(row['Capacity (MW)'])
            upper = capacity
            
            # 计算上下限（假设下限是最小输出率×容量，上限是容量）
            min_output_rate = float(row['Minimum output rate (%)']) / 100
            
            ramp = float(row['Maximum ramp rate (%/min)']) / 100
            max_start_time = float(row['Minimum on time (h)'])
            max_shut_time = float(row['Minimum off time (h)'])
            fixcost = float(row['Fixed cost\n(10^4CNY/(MW·year))'])
            outcost = float(row['Variable cost\n(10^4 CNY/MWh)'])
            startcost = float(row['Start cost\n(10^4 CNY/MWh)'])
            last_output = -1
            
            gen = Gen(gen_id, bus_id, type, region, capacity, upper, min_output_rate, ramp, max_start_time, max_shut_time, fixcost, outcost, startcost, last_output)
            gens.append(gen)
        except (ValueError, TypeError) as e:
            print(f"创建Gen对象时出错 (行 {index+1}): {e}")
            continue
    
    return gens

def create_storage_objects(storage_df, all_sheets):
    """创建Storage对象列表"""
    storages = []
    
    # 从Storage-Parameters表中获取效率信息
    storage_params_df = all_sheets.get('Storage-Parameters', pd.DataFrame())
    efficiency_map = {}
    
    if not storage_params_df.empty:
        for _, param_row in storage_params_df.iterrows():
            try:
                unit_type = int(param_row['Unit type'])
                efficiency = float(param_row['Eta (%)'])
                efficiency_map[unit_type] = efficiency / 100  # 转换为小数
            except:
                continue
    
    for index, row in storage_df.iterrows():
        try:
            # 处理Storage数据
            storage_id = int(row['Unit No.'])
            bus_id = int(row['Bus No.'])
            capacity = float(row['Capacity (MW)'])
            
            # 从映射中获取效率，如果没有则使用默认值
            ####### 需要修改
            unit_type = int(row['Unit Type'])
            efficiency = efficiency_map.get(unit_type, 0.9)  # 默认效率90%
            
            # 从Storage-Parameters表中获取储能时间
            storage_time = 0
            if not storage_params_df.empty:
                for _, param_row in storage_params_df.iterrows():
                    if int(param_row['Unit type']) == unit_type:
                        storage_time = float(param_row['Storage hour (h)'])
                        break
            
            soc = 0.5

            storage = Storage(storage_id, bus_id, capacity, efficiency, storage_time, soc)
            storages.append(storage)
        except (ValueError, TypeError) as e:
            # print(f"创建Storage对象时出错 (行 {index+1}): {e}")
            continue
    
    return storages

# 自动加载数据
load_data()

def load_case_data():
    return {
        'buses': buses,
        'lines': lines, 
        'gens': gens,
        'storages': storages
    }