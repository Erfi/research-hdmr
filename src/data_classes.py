from collections import namedtuple

DatasetInfo = namedtuple('DatasetInfo',
                         [
                             'data_range',
                             'primary_cut_center',
                             'varying_index',
                             'secondary_cut_center',
                             'fixed_index'
                         ],
                         defaults=(None,)*2)

TrainingSet = namedtuple('TrainingSet', ['X','Y'])

KeyInfo = namedtuple('KeyInfo', ['primary_cut_center', 'varying_index', 'secondary_cut_center', 'fixed_index'],
                     defaults=(None,)*2)
