import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from bokeh import models, plotting, io
from bokeh.models import LinearColorMapper, Slider, Div
from bokeh.layouts import row, column
from time import sleep
from itertools import cycle


color_mapper = LinearColorMapper(palette="Turbo256", low=-np.pi, high=np.pi)

# side of the box
L = 32.0
# initial density
rho_init = 1.0
# noise
eta = 0.15
r0 = 1.0
deltat = 1.0
factor = 0.5
v0 = r0/deltat*factor
N = int(rho_init*L**2)
pos = np.random.uniform(0, L, size=(N, 2))
orient = np.random.uniform(-np.pi, np.pi, size=N)
polarisation = [np.absolute(np.exp(1j*orient).mean())]
time = [0]

def animate():
    global orient, pos, polarisation, time
    N = pos.shape[0]

    tree = cKDTree(pos, boxsize=[L, L])
    dist = tree.sparse_distance_matrix(tree, max_distance=r0, output_type='coo_matrix')

    # important 3 lines: we evaluate a quantity for every column j
    data = np.exp(orient[dist.col]*1j)
    # construct a new sparse marix with entries in the same places ij of the dist matrix
    neigh = sparse.coo_matrix((data, (dist.row, dist.col)), shape=dist.get_shape())
    # and sum along the columns (sum over j)
    S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))

    orient = np.angle(S) + eta*np.random.uniform(-np.pi, np.pi, size=N)

    cos, sin = np.cos(orient), np.sin(orient)
    pos[:, 0] += cos*v0
    pos[:, 1] += sin*v0

    pos[pos > L] -= L
    pos[pos < 0] += L

    _pol = np.absolute(np.exp(1j*orient).mean())
    if len(time) < 100:
        time.append(time[-1] + 1)
        polarisation.append(_pol)
    else:
        time.append(time[-1] + 1)
        polarisation.append(_pol)
        time = time[1:]
        polarisation = polarisation[1:]

    return pos[:, 0], pos[:, 1], orient

x, y, angle = animate()
data = {'x': x, 'y': y, 'angle': angle}
timedata = {'time': time, 'pol': polarisation}
source = models.ColumnDataSource(data)
timesource = models.ColumnDataSource(timedata)

p = plotting.figure(
    title="Vicsek model simulator",
    tools=["save", "reset", "box_zoom"],
    plot_width=600, plot_height=600,
    x_range=(0, L),
    y_range=(0, L)

)
p.toolbar.logo = None
p.scatter(x="x", y="y",
          source=source,
          angle="angle",
          marker="dash",
          line_color={'field': 'angle', 'transform': color_mapper},
          # line_alpha="color",
          size=8,
          width=4.0,
          syncable=False
)

timeseries = plotting.figure(
    plot_width=300, plot_height=200,
    x_axis_label="Time",
    y_axis_label="Polarisation",
    y_range=(0, 1),
    # x_range=(0, L),
    # y_range=(0, L)
)
timeseries.toolbar_location = None
timeseries.scatter(x="time", y="pol",
                   source=timesource, marker='circle')

density_slider = Slider(start=0.1, end=3.5, value=rho_init, step=.1, title="Density")
noise_slider = Slider(start=0.01, end=1.0, value=0.1, step=.01, title="Noise")
speed_slider = Slider(start=0.02, end=5.0, value=1.0, step=.02, title="Speed")

def reset(attr, old, new):
    global orient, pos
    N = int(density_slider.value*L**2)
    print(" N", N)
    pos = np.random.uniform(0, L, size=(N, 2))
    orient = np.random.uniform(-np.pi, np.pi, size=N)

def update_noise(attr, old, new):
    global eta
    eta = noise_slider.value
    
def update_speed(attr, old, new):
    global v0
    v0 = speed_slider.value

def stream():
    x, y, angle = animate()
    data = {'x': x, 'y': y, 'angle': angle}
    timedata = {'time': time, 'pol': polarisation}
    source.data = data
    timesource.data = timedata

density_slider.on_change('value', reset)
noise_slider.on_change('value', update_noise)
speed_slider.on_change('value', update_speed)

controls = column(density_slider, noise_slider, speed_slider, timeseries, Div(
    text='by <a href="https://francescoturci.net" target="_blank">Francesco Turci </a>'))

io.curdoc().add_root(row(p, controls))
io.curdoc().add_periodic_callback(stream, 100)
io.curdoc().title = "Vicsek model simulator"
