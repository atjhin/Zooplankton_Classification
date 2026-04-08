"""
constants.py — Project-wide configuration values and taxonomy adjacency graphs.

Adjacency graphs:
    hier_adjacency_graph   : 3-level freshwater zooplankton taxonomy (Zooplankton-MNR dataset).
    whoi_adjacency_graph_l : 5-level marine diatom taxonomy based on phylogenetic lineage
                             (Plankton-WHOI dataset, long/deep hierarchy variant).
    whoi_adjacency_graph_s : 3-level marine diatom taxonomy based on morphological traits
                             (Plankton-WHOI dataset, short/shallow hierarchy variant).
"""

ZOOPLANKTON_CLASSES = [
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

import os
data_directory = os.environ.get('ZOOPLANKTON_DATA_DIR', 'data')
data_subdirectories = ['Processed Data']
SEED = 666            # Global reproducibility seed

MAX_CLASS_SIZE = 6000  # Maximum samples drawn per class during dataset construction
RESOLUTION = 64        # Target image resolution in pixels (H × W)

LEVELS = 3             # Number of non-root levels in the MNR hierarchy

# ---------------------------------------------------------------------------
# Zooplankton-MNR hierarchy (freshwater, Ontario Ministry of Natural Resources)
# 18 nodes | 3 levels | 12 leaf classes
# ---------------------------------------------------------------------------
hier_adjacency_graph = {
    'root': ['Zoop-yes', 'Zoop-No'],
    
    'Zoop-yes': ['Copepoda', 'Cladocera', 'Rotifer'],
    'Zoop-No': ['Bubbles', 'Exoskeleton', 'Fiber', 'Plant_Matter'],
    
    'Copepoda': ['Cyclopoid', 'Calanoid', 'Harpacticoid', 'Nauplius_Copepod'],
    'Cladocera': ['Bosminidae', 'Daphnia'],

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

# ---------------------------------------------------------------------------
# Plankton-WHOI hierarchy — phylogenetic lineage variant
# 27 nodes | 5 levels | 14 leaf classes
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Plankton-WHOI hierarchy — morphological traits variant
# 21 nodes | 3 levels | 16 leaf classes
# ---------------------------------------------------------------------------
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