import os, sys
sys.path.extend([os.environ[var] for var in ['ARES', 'PERSES']])
from perses.analysis import ModelSet


if len(sys.argv) == 1:
    raise NotImplementedError("CompressModelSet.py script must be called " +\
                              "like 'ipython " +\
                              "$PERSES/perses/util/CompressModelSet.py " +\
                              "[prefix] [options]' where [prefix] is the " +\
                              "absolute path prefix to the data.")
else:
    prefix = sys.argv[1]
    anl = ModelSet(prefix)
    kwargs = {}
    eq_strs = sys.argv[2:]
    for eq_str in eq_strs:
        name, val_str = eq_str.split('=')
        kwargs[name] = eval(val_str)
    anl.remove_effective_burnin(**kwargs)
