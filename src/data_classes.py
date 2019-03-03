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

