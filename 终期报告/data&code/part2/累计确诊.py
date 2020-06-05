import shapefile
from matplotlib import pyplot as plt
border_shape=shapefile.Reader("C:/Users/Administrator.PC-20190816XJBQ/Desktop/china.shp")
border_shape=shapefile.Reader("C:/Users/Administrator.PC-20190816XJBQ/Desktop/china_nine_dotted_line.shp")
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from pathlib import Path
from pathlib import Path
import csv

def outline(ll_lons,ll_lat,ur_lon,ur_lat,lat_0,lon_0) :
    m=Basemap(llcrnrlon=ll_lons,llcrnrlat=ll_lat,urcrnrlon=ur_lon,urcrnrlat=ur_lat,
             projection='lcc',lat_0=lat_0,lon_0=lon_0)
    m.readshapefile('china','china',drawbounds=True)
    m.readshapefile('china_nine_dotted_line','dashed9',drawbounds=True)
    return m

def colors(val) :
    if val<100 :
        return '#ffd7bb'
    elif val<500 :
        return '#ffad83'
    elif val <1000 :
        return '#f2684f'
    elif val <10000 :
        return '#d02327'
    else :
        return '#6f151c'

with Path('2020.4.30.csv').open() as p :
    reader=csv.DictReader(p)
    data={item['Province'][:2] : int(item['Confirmed']) for item in reader}

def draw():
    fig=plt.figure(figsize=(16,8))
    addcolar(outline(78,14,145,51,33,100))
    fig.add_axes([0.64,0.139,0.14,0.16])
    addcolar(outline(105,0,125,25,12,115))
    plt.show()

def addcolar(m) :
    ax=plt.gca()
    for state,info in zip(m.china,m.china_info) :
        name=info['OWNER'][:2]
        poly = Polygon(state,facecolor=colors(data[name]))
        ax.add_patch(poly)
draw()
