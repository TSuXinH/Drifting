import h5py
from scipy import io as scio


path = './data_alter/brain_region/atlas_top_projection.mat'
data = scio.loadmat(path)
keys = list(data.keys())

acronyms = data['acronyms'].reshape(-1)  # abbr
atlas_outline = data['atlas_outline']  # map, shape: [456, 528]
clean_atlas_outline = data['clean_atlas_outline']
cortex_outline = data['cortex_outline']  # cortex
clean_cortex_outline = data['clean_cortex_outline']
ids = data['ids'].reshape(-1)  # all kinds of string numbers as the identifier of different brain regions
names = data['names'].reshape(-1)  # the specific name of different brain regions
parents = data['parents'].reshape(-1)  # todo: unknown
top_projection = data['top_projection']  # the distribution of different brain regions under projection.

x_list = []
for x in range(len(names)):
    if 'hippocampus' in acronyms[x].item().lower():
        x_list.append(x)

