from perses import beam, foregrounds, models, simulations, util

class_names = ['perses.models.Full21cmModel.Full21cmModel']
# init not included in magic_names because __init__ is automatically documented
# hash not included because it appears automatically
magic_names = ['new', 'del', 'repr', 'str', 'bytes', 'format', 'lt', 'le',\
    'eq', 'ne', 'gt', 'ge', 'bool', 'getattr', 'getattribute', 'setattr',\
    'delattr', 'dir', 'get', 'set', 'delete', 'set_name', 'slots',\
    'init_subclass', 'class_getitem', 'call', 'len', 'length_hint', 'getitem',\
    'setitem', 'delitem', 'missing', 'iter', 'reversed', 'contains', 'add',\
    'sub', 'mul', 'matmul', 'truediv', 'floordiv', 'div', 'mod', 'divmod',\
    'pow', 'lshift', 'rshift', 'and', 'or', 'xor', 'radd', 'rsub', 'rmul',\
    'rmatmul', 'rtruediv', 'rfloordiv', 'rdiv', 'rmod', 'rdivmod', 'rpow',\
    'rlshift', 'rrshift', 'rand', 'ror', 'rxor', 'iadd', 'isub', 'imul',\
    'imatmul', 'itruediv', 'ifloordiv', 'idiv', 'imod', 'ipow', 'ilshift',\
    'irshift', 'iand', 'ior', 'ixor', 'neg', 'pos', 'abs', 'invert',\
    'complex', 'int', 'float', 'index', 'round', 'trunc', 'floor', 'ceil',\
    'next', 'enter', 'exit', 'await', 'aiter', 'anext', 'aenter', 'aexit']
__pdoc__ = {}
for magic_name in magic_names:
    for class_name in class_names:
        __pdoc__['{0!s}.__{1!s}__'.format(class_name, magic_name)] = True
