ZOOPLANKTON_CLASSES = [
    #  'Debris',
     'Bubbles',
     'Exoskeleton',
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
    'Zoop-No': ['Bubbles', 'Exoskeleton', 'Fiber'],
    
    'Copepoda': ['Cyclopoid', 'Calanoid', 'Harpacticoid', 'Nauplius_Copepod'],
    'Cladocera': ['Bosminidae', 'Daphnia'],
    # 'Debris': [],
    'Rotifer': [],  
    'Bubbles': [],  
    'Exoskeleton': [],  
    'Fiber': ['Fiber_Hairlike', 'Fiber_Squiggly'],
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
      'root': ['Bacilllariophytina', 'Coscinodiscophytina'],                                                                                                                            
                                                                                                                                                                                        
      'Bacilllariophytina': ['Bacilliorhycaea', 'Mediophycaea'],                                                                                                                        
      'Coscinodiscophytina': ['Coscinodiscophyceae'],                                                                                                                                   
                                                                                                                                                                                        
      'Bacilliorhycaea': ['Bacillariophycidae', 'Fragilariophycidae'], 
      'Mediophycaea': ['Chaetocerothophycidae', 'Thalassiosirophycidae'],
      'Coscinodiscophyceae': ['Corethopycidae', 'Rhyzosoleniophycidae'],

      'Bacillariophycidae': ['Bacillariales'],
      'Fragilariophycidae': ['Thalassionematales'],
      'Chaetocerothophycidae': ['Chaetocerotales', 'Hemiaulales'],
      'Thalassiosirophycidae': ['Lethodesmiales', 'Thalassiosoreles'],
      'Corethopycidae': ['Corethales'],
      'Rhyzosoleniophycidae': ['Rhyzosoleniophycidae_order'],

      'Bacillariales': ['Bacillariaceae'],
      'Thalassionematales': ['Thalassionematecaea'],
      'Chaetocerotales': ['Chaetocerotacae', 'Leptocylindaceae'],
      'Hemiaulales': ['Hemiaulaceae'],
      'Lethodesmiales': ['Lithodesmiaceae'],
      'Thalassiosoreles': ['Skeletonemataceae', 'Thalassiosiraceae'],
      'Corethales': ['Corethraceae'],
      'Rhyzosoleniophycidae_order': ['Rhizosoleniaceae'],

      'Bacillariaceae': ['Cylindrotheca', 'Pseudonitzchia'],
      'Thalassionematecaea': ['Thalassiomema'],
      'Chaetocerotacae': ['Chaetoceros'],
      'Leptocylindaceae': ['Leptocylindrus'],
      'Hemiaulaceae': ['Cerataulina', 'Eucampia'],
      'Lithodesmiaceae': ['Ditylum'],
      'Skeletonemataceae': ['Skeletonema'],
      'Thalassiosiraceae': ['Thalassiosira'],
      'Corethraceae': ['Corethon'],
      'Rhizosoleniaceae': ['Dactyliosolem', 'Guinardia', 'Rhizosolenia'],

      'Cylindrotheca': [],
      'Pseudonitzchia': [],
      'Thalassiomema': [],
      'Chaetoceros': [],
      'Leptocylindrus': [],
      'Cerataulina': [],
      'Eucampia': [],
      'Ditylum': [],
      'Skeletonema': [],
      'Thalassiosira': [],
      'Corethon': [],
      'Dactyliosolem': [],
      'Guinardia': [],
      'Rhizosolenia': [],
  }

whoi_adjacency_graph_s = {                                                                                                                                                   
      'root': ['Colonial', 'Unicellular'],                                                                                                                                              
                                                                                                                                                                                        
      'Colonial': ['C-Spines', 'C-NoSpines'],                                                                                                                                           
      'Unicellular': ['U-Spines', 'U-NoSpines'],                                                                                                                                        
                                                                                                                                                                                      
      'C-Spines': ['Chaetoceros', 'Lauderia', 'Asterionellopsis'],
      'C-NoSpines': ['Pseudonitzchia', 'Leptocylindrus', 'Eucampia', 'Skeletonema',
                     'Dactylosolen', 'Thalassiosira', 'Guinardia', 'Cerataulina'],
      'U-Spines': ['Corethon', 'Ditylum'],
      'U-NoSpines': ['Cylindrotheca', 'Coscinodiscus', 'Ephemera'],

      'Chaetoceros': [],
      'Lauderia': [],
      'Asterionellopsis': [],
      'Pseudonitzchia': [],
      'Leptocylindrus': [],
      'Eucampia': [],
      'Skeletonema': [],
      'Dactylosolen': [],
      'Thalassiosira': [],
      'Guinardia': [],
      'Cerataulina': [],
      'Corethon': [],
      'Ditylum': [],
      'Cylindrotheca': [],
      'Coscinodiscus': [],
      'Ephemera': [],
  }