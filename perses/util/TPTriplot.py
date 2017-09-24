import os, sys
sys.path.extend([os.environ[var] for var in ['ARES', 'PERSES']])
import numpy as np
import matplotlib.pyplot as pl
from perses.analysis import ModelSet

prefix = sys.argv[1]
tps = []
kwargs = {'bins': 10, 'color_by_like': [0.68, 0.95]}
for arg in sys.argv[2:]:
    equality = arg.split('=')
    if len(equality) == 1:
        tps.append(equality[0])
    elif len(equality) == 2:
        arg, val = equality
        kwargs[arg] = eval(val)
    else:
        raise ValueError('Arguments were not supplied as expected. ' +\
                         'For turning points, just add the letter of ' +\
                         'the turning point to the sys.argv. For ' +\
                         'keyword arguments to pass to anl.TrianglePlot, ' +\
                         'add arg=val to sys.argv.')
if not tps:
    raise ValueError("No turning points were given so none can be plotted.")
zs = [('z_' + tp) for tp in tps]

anl = ModelSet(prefix)

if 'include_checkpoints' in kwargs:
    anl.include_checkpoints = kwargs['include_checkpoints']
    del kwargs['include_checkpoints']
else:
    anl.include_checkpoints = anl.last_n_checkpoints(5)
if 'elements' in kwargs:
    start, stop, skim = kwargs['elements']
    elements = range(start, stop, skim)
    del kwargs['elements']
else:
    elements = np.arange(len(anl.logL))
if 'clobber' in kwargs:
    clobber = kwargs['clobber']
    del kwargs['clobber']
else:
    clobber = False
anl.save_turning_points_from_elements(elements=elements, which=tps,\
    clobber=clobber)

anl.pf['blob_names'] = [zs]
anl.pf['blob_ivars'] = [None]
anl.pf['blob_funcs'] = [None]
anl._parse_blobs()

anl.TrianglePlot(zs, **kwargs)

pl.show()

