import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from bokeh import models, plotting, io
from bokeh.models import LinearColorMapper, Slider, Div, Button
from bokeh.layouts import row, column
from time import sleep
from itertools import cycle


def animate(pos, orient, seed=-1):
    N = pos.shape[0]

    tree = cKDTree(pos, boxsize=[L, L])
    dist = tree.sparse_distance_matrix(tree, max_distance=r0, output_type='coo_matrix')

    # important 3 lines: we evaluate a quantity for every column j
    data = np.exp(orient[dist.col]*1j)
    # construct a new sparse marix with entries in the same places ij of the dist matrix
    neigh = sparse.coo_matrix((data, (dist.row, dist.col)), shape=dist.get_shape())
    # and sum along the columns (sum over j)
    S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))

    np.random.seed(seed)
    orient[:] = np.angle(S) + eta*np.random.uniform(-np.pi, np.pi, size=N)
    cos, sin = np.cos(orient), np.sin(orient)
    pos[:, 0] += cos*v0
    pos[:, 1] += sin*v0
    pos[pos > L] -= L
    pos[pos < 0] += L

def polarisation(angle):
    return np.absolute(np.exp(1j*angle).mean())

color_mapper = LinearColorMapper(palette="Turbo256", low=-np.pi, high=np.pi)

# side of the box
L = 32.0
# initial density
rho_init = 1.0
# noise
eta = 0.15
r0 = 1.0
deltat = 1.0
factor = 0.25
# TODO: speed should update r0 or factor and v0 recalculate
v0 = r0/deltat*factor
N = int(rho_init*L**2)
pos_1 = np.random.uniform(0, L, size=(N, 2))
pos_2 = pos_1.copy()
orient_1 = np.random.uniform(-np.pi, np.pi, size=N)
orient_2 = orient_1.copy()
polar_1 = [np.absolute(np.exp(1j*orient_1).mean())]
polar_2 = [np.absolute(np.exp(1j*orient_2).mean())]
time = [0]


source_1 = models.ColumnDataSource({'x': pos_1[:, 0], 'y': pos_1[:, 1], 'angle': orient_1})
source_2 = models.ColumnDataSource({'x': pos_2[:, 0], 'y': pos_2[:, 1], 'angle': orient_2})
timesource = models.ColumnDataSource({'time': time, 'pol_1': polar_1, 'pol_2': polar_2})

plot = plotting.figure(
    title="Modello di Vicsek",
    tools=["save", "reset", "box_zoom"],
    plot_width=600, plot_height=600,
    x_range=(0, L),
    y_range=(0, L)

)
plot.toolbar.logo = None
plot.scatter(x="x", y="y",
             source=source_1,
             angle="angle",
             marker="dash",
             line_color={'field': 'angle', 'transform': color_mapper},
             # line_alpha="color",
             size=8,
             width=4.0,
             syncable=False
)

plot_2 = plotting.figure(
    title="Replica",
    tools=["save", "reset", "box_zoom"],
    plot_width=600, plot_height=600,
    x_range=(0, L),
    y_range=(0, L)
)

plot_2.toolbar.logo = None
plot_2.scatter(x="x", y="y",
               source=source_2,
               angle="angle",
               marker="dash",
               line_color={'field': 'angle', 'transform': color_mapper},
               # line_alpha="color",
               size=8,
               width=4.0,
               syncable=False
)

timeseries = plotting.figure(
    plot_width=400, plot_height=300,
    x_axis_label="Tempo",
    y_axis_label="Velocità media",
    y_range=(0, 1),
)
timeseries.toolbar_location = None
# TODO: this does not work
timeseries.line("time", "pol_1", source=timesource, color='blue')
timeseries.line("time", "pol_2", source=timesource, color='red')

density_slider = Slider(start=0.1, end=3.5, value=rho_init, step=.1, title="Densità")
noise_slider = Slider(start=0.01, end=1.0, value=0.1, step=.01, title="Rumore")
speed_slider = Slider(start=0.02, end=5.0, value=1.0, step=.02, title="Velocità")

def reset(attr, old, new):
    global orient_1, pos_1, orient_2, pos_2
    N = int(density_slider.value*L**2)
    print(" N", N)
    pos_1 = np.random.uniform(0, L, size=(N, 2))
    pos_2 = pos_1.copy()
    orient_1 = np.random.uniform(-np.pi, np.pi, size=N)
    orient_2 = orient_1.copy()

def update_noise(attr, old, new):
    global eta
    eta = noise_slider.value
    
def update_speed(attr, old, new):
    global v0
    v0 = speed_slider.value

def stream():
    global time, polar_1, polar_2, source_1
    animate(pos_1, orient_1, time[-1])
    animate(pos_2, orient_2, time[-1])
    time.append(time[-1] + 1)
    polar_1.append(polarisation(orient_1))
    polar_2.append(polarisation(orient_2))
    if time[-1] > 100:
        time = time[1:]
        polar_1 = polar_1[1:]
        polar_2 = polar_2[1:]
    source_1.data = {'x': pos_1[:, 0], 'y': pos_1[:, 1], 'angle': orient_1}
    source_2.data = {'x': pos_2[:, 0], 'y': pos_2[:, 1], 'angle': orient_2}
    timesource.data = {'time': time, 'pol_1': polar_1, 'pol_2': polar_2}

density_slider.on_change('value', reset)
noise_slider.on_change('value', update_noise)
speed_slider.on_change('value', update_speed)

callback_id = None
def run():
    global callback_id
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        callback_id = io.curdoc().add_periodic_callback(stream, 100)
    else:
        button.label = '► Play'
        io.curdoc().remove_periodic_callback(callback_id)

button = Button(label='► Play', width=60)
button.on_event('button_click', run)

def clone():
    pos_2[:, :] = pos_1.copy()
    orient_2[:] = orient_1.copy()
    source_1.data = {'x': pos_1[:, 0], 'y': pos_1[:, 1], 'angle': orient_1}
    source_2.data = {'x': pos_2[:, 0], 'y': pos_2[:, 1], 'angle': orient_2}

button_clone = Button(label='Clona la replica', width=60)
button_clone.on_event('button_click', clone)

def perturb():
    orient_1[0] += 1e-2
    source_1.data = {'x': pos_1[:, 0], 'y': pos_1[:, 1], 'angle': orient_1}
    source_2.data = {'x': pos_2[:, 0], 'y': pos_2[:, 1], 'angle': orient_2}

button_perturb = Button(label='Perturba la replica', width=60)
button_perturb.on_event('button_click', perturb)

controls = column(button, button_clone, button_perturb, density_slider, noise_slider, speed_slider, timeseries, Div(
    text='by <a href="https://francescoturci.net" target="_blank">Francesco Turci </a>'))

# layout = layout([
#     [plot],
#     [slider, button],
# ], sizing_mode='scale_width')

io.curdoc().add_root(row(plot, plot_2, controls))
io.curdoc().title = "Vicsek model"

