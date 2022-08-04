import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from GridResilience.Environment import *
import numpy as np
import geopandas as gpd

if __name__ == "__main__":
    grid = GridCase()
    met_data = pd.read_csv('c:/users/yunqi/desktop/极端灾害.csv', header=0, encoding='gbk')
    grid.update_meteorological_data(met_data)
    grid_lon = np.linspace(grid.extend_attr["met"]["经纬度"]["经度"].min(), grid.extend_attr["met"]["经纬度"]["经度"].max(), 400)
    grid_lat = np.linspace(grid.extend_attr["met"]["经纬度"]["纬度"].min(), grid.extend_attr["met"]["经纬度"]["纬度"].max(), 400)

    OK = OrdinaryKriging(grid.extend_attr["met"]["经纬度"]["经度"], grid.extend_attr["met"]["经纬度"]["纬度"],
                         grid.extend_attr["met"]["温度"][0],
                         variogram_model='gaussian', nlags=6)
    z1, ss1 = OK.execute('grid', grid_lon, grid_lat)
    xgrid, ygrid = np.meshgrid(grid_lon, grid_lat)
    df_grid = pd.DataFrame(dict(long=xgrid.flatten(), lat=ygrid.flatten()))
    df_grid["Krig_gaussian"] = z1.flatten()
    import plotly.express as px
    import plotly.io as pio
    import matplotlib.pyplot as plt

    plt.style.use(['science', 'no-latex', 'std-colors'])
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 22
    plt.rcParams['axes.unicode_minus'] = False


    # plt.subplot(121)
    # for i in range(len(met_data)):
    #     plt.scatter(met_data[''])
    #
    # plt.subplot(122)
    def find_closest(A, target):
        # A must be sorted
        idx = A.searchsorted(target)
        idx = np.clip(idx, 1, len(A) - 1)
        left = A[idx - 1]
        right = A[idx]
        idx -= target - left < right - target
        return idx


    plt.matshow(z1)
    for i in range(11):
        lat = met_data['经度'].iloc[i]
        lng = met_data['纬度'].iloc[i]
        x = find_closest(xgrid[0, :], lat)
        y = find_closest(ygrid[:, 0], lng)
        print(f'{x} {y}')
        plt.scatter(y, x, marker='o', s=50)

    #
    # cl.plot(column='')
