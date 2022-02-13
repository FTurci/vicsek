import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
from bokeh import models, plotting, io
from bokeh.models import LinearColorMapper

color_mapper = LinearColorMapper(palette="Turbo256", low=-np.pi, high=np.pi)
#
L = 32.0
rho = 4.0
N = int(rho*L**2)
print(" N",N)

r0 = 1.0
deltat = 1.0
factor =0.5
v0 = r0/deltat*factor
iterations = 10000
eta = 0.15


pos = np.random.uniform(0,L,size=(N,2))
orient = np.random.uniform(-np.pi, np.pi,size=N)

def animate():
    global orient, pos
    tree = cKDTree(pos,boxsize=[L,L])
    dist = tree.sparse_distance_matrix(tree, max_distance=r0,output_type='coo_matrix')

    #important 3 lines: we evaluate a quantity for every column j
    data = np.exp(orient[dist.col]*1j)
    # construct  a new sparse marix with entries in the same places ij of the dist matrix
    neigh = sparse.coo_matrix((data,(dist.row,dist.col)), shape=dist.get_shape())
    # and sum along the columns (sum over j)
    S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))


    orient = np.angle(S)+eta*np.random.uniform(-np.pi, np.pi, size=N)


    cos, sin= np.cos(orient), np.sin(orient)
    pos[:,0] += cos*v0
    pos[:,1] += sin*v0

    pos[pos>L] -= L
    pos[pos<0] += L

    # qv.set_offsets(pos)
    # qv.set_UVC(cos, sin,orient)
    return pos[:,0], pos[:,1], orient
#
# FuncAnimation(fig,animate,np.arange(1, 200),interval=1, blit=True)
# plt.show()


from bokeh import models, plotting, io
import pandas as pd
from time import sleep
from itertools import cycle
import numpy as np



x,y,angle= animate()
data= {'x':x,
'y':y,'angle':angle
}


source = models.ColumnDataSource(data)

p = plotting.figure(
    # x_axis_label="Date",
    # y_axis_label="New Cases",
    plot_width=600, plot_height=600,
     # x_axis_type="datetime",
     # tools=["hover", "wheel_zoom"],
     x_range=(0,L),
     y_range=(0,L)
)

p.scatter(x="x", y="y",
       source=source,
       angle="angle",
       marker="dash",
       line_color={'field': 'angle', 'transform': color_mapper},
       # line_alpha="color",
       size=8,
       width=4.0
       )

io.curdoc().add_root(p)
# index_generator = cycle(range(len(california_covid_data.index)))

def stream():
    x,y,angle = animate()
    data= {'x':x,
    'y':y,'angle':angle,
    }

    source.data = data

io.curdoc().add_periodic_callback(stream, 10)
