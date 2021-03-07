import numpy as np
from bokeh.plotting import figure, show, output_file

output_file('templates/wave_plot.html', title='Wave Plot',)
p = figure(plot_width=800, plot_height=600)

n = 100
x = np.arange(0, n, 1)
y = np.random.random(n)

p.line(x, y, line_width=3)
#show(p)
