import time

import pandapower
import pandapower as pp
import numpy as np
import pandapower.converter as pc
from pandapower.networks import case33bw
from pypower.api import case14
from GridResilience.Environment.case_pandas import GridCase
from GridResilience.Environment.idx_cpsbrh import *
from GridResilience.Environment.idx_cpsbus import *
from copy import deepcopy


def type_converter(arr, column=1):
    type_changer = np.frompyfunc(lambda x: LOAD if x in {1} else SOURCE, 1, 1)
    arr[:, column] = type_changer(arr[:, column])
    return arr


def sorted_from_json(path):
    net = pp.from_json(path)
    for elm in pp.pp_elements():
        net[elm].sort_index(inplace=True)
    return net


def ppc_to_gc(pp_case: dict) -> GridCase:
    bus = pp_case["bus"]
    bus_types = bus[:, 1]
    pos = np.zeros(np.size(bus, axis=0))
    bus = np.insert(bus, 1, pos, axis=1)
    bus = np.insert(bus, 1, pos, axis=1)

    bus = type_converter(bus, column=BUS_TYPE)
    bus = np.insert(bus, PQV_TYPE, bus_types, axis=1)
    bus = np.insert(bus, VALUE, np.ones(np.size(bus, axis=0)), axis=1)[:, :17]
    brh = pp_case["branch"]
    brh = np.insert(brh, 0, np.arange(start=100000, stop=100000 + np.size(brh, axis=0)), axis=1)[:, :14]
    ls = np.ones((np.size(brh, axis=0), 1))
    internal_case = GridCase().from_array(brh, bus, ls)
    internal_case.baseMVA = pp_case["baseMVA"]
    for key in pp_case:
        if key != "bus" and key != "branch" and key != "baseMVA":
            internal_case.extend_attr[key] = pp_case.get(key, None)
    return internal_case


def mpc_to_gc(mpc_file: str, f_hz=50, case_name_inside='mpc', validation=False) -> GridCase:
    return ppc_to_gc(pc.to_ppc(pc.from_mpc(mpc_file, f_hz, case_name_inside, validation)))


def pandapower_to_gc(pp_net: pandapower.pandapowerNet) -> GridCase:
    local_pp_net = deepcopy(pp_net)
    local_pp_net['line']['in_service'] = True
    return ppc_to_gc(pc.to_ppc(local_pp_net, init='flat'))


if __name__ == "__main__":
    import pandas as pd

    net = case33bw()
    grid = pandapower_to_gc(net)
    met_data = pd.read_csv('c:/users/yunqi/desktop/极端灾害.csv', header=0, encoding='gbk')
    grid.update_meteorological_data(met_data)
    cs_data = pd.read_csv('c:/users/yunqi/desktop/交通充电站.csv', header=0, encoding='gbk')
    grid.update_charging_station_data(cs_data)
