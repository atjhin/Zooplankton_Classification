ZOOPLANKTON_CLASSES = [
    #  'Debris',
     'Bubbles',
     'Exoskeleton',
     'Plant_Matter',
     'Fiber_Squiggly',
     'Fiber_Hairlike',
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

MAX_CLASS_SIZE = 6000
RESOLUTION = 64

LEVELS = 3

# coarse_names1 = ['Zoop-yes', 'Zoop-No']
# groups1 = [
#     [
#         'Copepoda','Cladocera','Bosminidae','Daphnia',
#         'Cyclopoid','Harpacticoid','Calanoid','Rotifer',
#         'Nauplius_Copepod'
#     ],
    
#     ['Debris','Bubbles','Exoskeleton','Fiber_Squiggly']
# ]

# coarse_names2 = [
#     'Copepoda', 'Cladocera','Rotifer','Bubbles', 'Exoskeleton',
#     'Fiber'
# ]
# groups2 = [
#     ['Copepoda','Cyclopoid','Calanoid','Harpacticoid','Nauplius_Copepod'],
#     ['Cladocera','Bosminidae','Daphnia'],
#     ['Rotifer'],
#     ['Bubbles'],
#     ['Exoskeleton'],
#     ['Fiber_Squiggly']
# ]

# coarse_names3 = [
#     'Cyclopoid','Calanoid','Harpacticoid','Nauplius_Copepod',
#     'Bosminidae','Daphnia','Rotifer','Bubbles','Exoskeleton','Fiber'
# ]
# groups3 = [
#     ['Cyclopoid'],
#     ['Calanoid'],
#     ['Harpacticoid'],
#     ['Nauplius_Copepod'],
#     ['Bosminidae'],
#     ['Daphnia'],
#     ['Rotifer'],
#     ['Bubbles'],
#     ['Exoskeleton'],
#     ['Fiber_Squiggly'] 
# ] 

# coarse_names = [coarse_names3,coarse_names2,coarse_names1]
# groups = [groups3, groups2, groups1]

hier_adjacency_graph = {
    'root': ['Zoop-yes', 'Zoop-No'],
    
    'Zoop-yes': ['Copepoda', 'Cladocera', 'Rotifer'],
    'Zoop-No': ['Bubbles', 'Exoskeleton', 'Fiber', 'Plant_Matter'],
    
    'Copepoda': ['Cyclopoid', 'Calanoid', 'Harpacticoid', 'Nauplius_Copepod'],
    'Cladocera': ['Bosminidae', 'Daphnia'],
    # 'Debris': [],
    'Fiber': ['Fiber_Hairlike', 'Fiber_Squiggly'],
    'Rotifer': [],  
    'Bubbles': [],  
    'Exoskeleton': [],  
    'Plant_Matter': [],
    'Fiber_Squiggly': [],
    'Fiber_Hairlike': [],
    'Cyclopoid': [],  
    'Calanoid': [],  
    'Harpacticoid': [],  
    'Nauplius_Copepod': [],  
    'Bosminidae': [],  
    'Daphnia': []  
}

whoi_adjacency_graph_l = {
      'root': ['Bacilllariophytina', 'Coscinodiscophyceae'],

      'Bacilllariophytina': ['Bacilliorhycaea', 'Mediophycaea'],
      'Coscinodiscophyceae': ['Corethron', 'Rhizosoleniaceae'],

      'Bacilliorhycaea': ['Bacillariaceae', 'Thalassionema'],
      'Mediophycaea': ['Chaetocerothophycidae', 'Thalassiosirophycidae'],

      'Chaetocerothophycidae': ['Chaetocerotales', 'Hemiaulaceae'],
      'Thalassiosirophycidae': ['Ditylum', 'Thalassiosoreles'],

      'Chaetocerotales': ['Chaetoceros', 'Leptocylindrus'],
      'Hemiaulaceae': ['Cerataulina', 'Eucampia'],
      'Thalassiosoreles': ['Skeletonema', 'Thalassiosira'],

      'Bacillariaceae': ['Cylindrotheca', 'Pseudonitzschia'],
      'Rhizosoleniaceae': ['Dactyliosolen', 'Guinardia', 'Rhizosolenia'],

      'Cylindrotheca': [],
      'Pseudonitzschia': [],
      'Thalassionema': [],
      'Chaetoceros': [],
      'Leptocylindrus': [],
      'Cerataulina': [],
      'Eucampia': [],
      'Ditylum': [],
      'Skeletonema': [],
      'Thalassiosira': [],
      'Corethron': [],
      'Dactyliosolen': [],
      'Guinardia': [],
      'Rhizosolenia': [],
  }

whoi_adjacency_graph_s = {
      'root': ['Colonial', 'Unicellular'],

      'Colonial': ['C-Spines', 'C-NoSpines'],
      'Unicellular': ['U-Spines', 'U-NoSpines'],

      'C-Spines': ['Chaetoceros', 'Lauderia', 'Asterionellopsis'],
      'C-NoSpines': ['Pseudonitzschia', 'Leptocylindrus', 'Eucampia', 'Skeletonema',
                     'Dactyliosolen', 'Thalassiosira', 'Guinardia', 'Cerataulina'],
      'U-Spines': ['Corethron', 'Ditylum'],
      'U-NoSpines': ['Cylindrotheca', 'Coscinodiscus', 'Ephemera'],

      'Chaetoceros': [],
      'Lauderia': [],
      'Asterionellopsis': [],
      'Pseudonitzschia': [],
      'Leptocylindrus': [],
      'Eucampia': [],
      'Skeletonema': [],
      'Dactyliosolen': [],
      'Thalassiosira': [],
      'Guinardia': [],
      'Cerataulina': [],
      'Corethron': [],
      'Ditylum': [],
      'Cylindrotheca': [],
      'Coscinodiscus': [],
      'Ephemera': [],
  }