ZOOPLANKTON_CLASSES = [
     'Debris',
     'Bubbles',
     'Exoskeleton',
     'Fiber_Squiggly',
     'Copepoda',
     'Calanoid',
     'Cyclopoid',
     'Harpacticoid',
     'Cladocera',
     'Bosminidae',
     'Daphnia',
     'Rotifer',
     'Nauplius_Copepod'
]

data_directory = '/Users/alexandermichaeltjhin/Everything/Repos/Zooplankton/data'
data_subdirectories = ['Processed Data']
SEED = 666

MAX_CLASS_SIZE = 10000
RESOLUTION = 64

LEVELS = 3

coarse_names1 = ['Zoop-yes', 'Zoop-No']
groups1 = [
    [
        'Copepoda','Cladocera','Bosminidae','Daphnia',
        'Cyclopoid','Harpacticoid','Calanoid','Rotifer',
        'Nauplius_Copepod'
    ],
    
    ['Debris','Bubbles','Exoskeleton','Fiber_Squiggly']
]

coarse_names2 = [
    'Copepoda', 'Cladocera','Rotifer','Bubbles', 'Exoskeleton',
    'Fiber'
]
groups2 = [
    ['Copepoda','Cyclopoid','Calanoid','Harpacticoid','Nauplius_Copepod'],
    ['Cladocera','Bosminidae','Daphnia'],
    ['Rotifer'],
    ['Bubbles'],
    ['Exoskeleton'],
    ['Fiber_Squiggly']
]

coarse_names3 = [
    'Cyclopoid','Calanoid','Harpacticoid','Nauplius_Copepod',
    'Bosminidae','Daphnia','Rotifer','Bubbles','Exoskeleton','Fiber'
]
groups3 = [
    ['Cyclopoid'],
    ['Calanoid'],
    ['Harpacticoid'],
    ['Nauplius_Copepod'],
    ['Bosminidae'],
    ['Daphnia'],
    ['Rotifer'],
    ['Bubbles'],
    ['Exoskeleton'],
    ['Fiber_Squiggly'] 
] 

coarse_names = [coarse_names3,coarse_names2,coarse_names1]
groups = [groups3, groups2, groups1]

hier_adjacency_graph = {
    'root': ['Zoop-yes', 'Zoop-No'],
    
    'Zoop-yes': ['Copepoda', 'Cladocera', 'Rotifer'],
    'Zoop-No': ['Debris', 'Bubbles', 'Exoskeleton', 'Fiber_Squiggly'],
    
    'Copepoda': ['Cyclopoid', 'Calanoid', 'Harpacticoid', 'Nauplius_Copepod'],
    'Cladocera': ['Bosminidae', 'Daphnia'],
    'Debris': [],
    'Rotifer': [],  
    'Bubbles': [],  
    'Exoskeleton': [],  
    'Fiber_Squiggly': [],
    
    'Cyclopoid': [],  
    'Calanoid': [],  
    'Harpacticoid': [],  
    'Nauplius_Copepod': [],  
    'Bosminidae': [],  
    'Daphnia': []  
}