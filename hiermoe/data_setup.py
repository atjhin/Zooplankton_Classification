import os
import random
from torch.utils.data import Dataset, Subset, DataLoader, SequentialSampler, WeightedRandomSampler, random_split
import torch 
import numpy as np 
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from collections import Counter
from extra_functions import set_seed
from hierarchy import Hierarchy

import copy
import torch.nn.functional as F


class ImageDataset(Dataset):

    """
    A custom PyTorch Dataset for loading and preprocessing image data from a directory
    where each subfolder represents a class.

    This class handles:
    - Class-wise and random sampling with optional size limits
    - Optional image transforms for data augmentation
    - Preprocessing and setup for imbalanced class handling

    Args:
        data_directory (str): Path to the root dataset directory. Each subdirectory should represent a class or another subdirectory to check.
        data_subdirectories (list of str, optional): Subdirectories with additional images, each sub-subdirectory should represent a class.
        class_names (list, optional): List of class names to include. If None, all subdirectories are included.
        class_sizes (list, optional): Number of samples to include per class. If None, uses `max_class_size` for all.
        class_ids (list, optional): Numeric ID for each class (aligned with `class_names`).
        max_class_size (int, optional): Default maximum number of samples to draw per class. Defaults to 10,000.
        image_resolution (int, optional): Final size (height and width) to resize images to. Defaults to 28.
        image_transforms (callable, optional): Image transformations (e.g., data augmentations) to apply. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 666.

    Attributes:
        data_directory (str): Path to the dataset root directory.
        data_subdirectories (list of str): Subdirectories with additional images.
        seed (int): Random seed used for sampling.
        class_names (list): Sorted list of class names included in the dataset.
        class_sizes (torch.Tensor): Tensor of the actual sampled size per class.
        class_ids (list): Numeric ID for each class (aligned with `class_names`).
        image_paths (list): List of file paths to all sampled images.
        labels (list): List of numeric class IDs corresponding to each image.
        image_resolution (int): Size to which each image is resized.
        image_transforms (callable or None): Image transformations applied during training or inference.
        format_file (str): Format of images files. Default: '.tif' 
    """

    def __init__(self, data_directory, data_subdirectories: list = None, class_names: list = None, 
                 class_sizes: list = None, class_ids: list = None, max_class_size: int = 10000, 
                 image_resolution: int = 28, image_transforms = None, seed: int = 666,
                 format_file = '.tif'):
        
        self.data_directory = data_directory
        self.data_subdirectories = ['']
        self.seed = seed
        self.class_names = class_names

        set_seed(seed)

        # Additional subdirectories to check
        if data_subdirectories is not None:
            self.data_subdirectories.extend(data_subdirectories)

        # Specify subset of classes to consider; all classes considered if None
        if class_names is None:
            class_names = sorted(os.listdir(self.data_directory))

        # Specify initial number of samples to consider per class; max if None
        if class_sizes is None:
            class_sizes = [max_class_size] * len(class_names)

        # Specify numeric class ID/index per class; in alphabetical order if None
        if class_ids is None:
            class_ids = list(range(len(self.class_names)))
        
        self.class_names, self.class_sizes, self.class_ids = map(
            list, zip(*sorted(zip(class_names, class_sizes, class_ids)))
        )
               
        # Iterate through each class and sample .tif images only; append paths and labels
        self.image_paths = []
        self.labels = []

        for class_id, class_name in zip(self.class_ids, self.class_names):
            
            # Retrieve all image paths across directories for specified class
            class_paths = []

            for data_subdirectory in self.data_subdirectories:

                class_directory = os.path.join(data_directory, data_subdirectory, class_name)

                if os.path.isdir(class_directory):
                    class_paths.extend(
                        [os.path.join(class_directory, filename) for filename in os.listdir(class_directory)]
                    )

            # Determine new class size and sample images, only include .tif files
            class_idx = class_ids.index(class_id)
            new_class_size = min(self.class_sizes[class_idx], len(class_paths))

            random.seed(self.seed)
            sampled_paths = random.sample(class_paths, new_class_size)
            
            for image_path in sampled_paths:
                if image_path.lower().endswith(format_file):
                    try:
                        with Image.open(image_path) as img:
                            img.verify()
                        self.image_paths.append(image_path)
                    except (UnidentifiedImageError, OSError, ValueError):
                        new_class_size -= 1
                else:
                    new_class_size -= 1

            self.class_sizes[class_idx] = new_class_size
            self.labels.extend([class_name] * new_class_size)
        
        # Other class initializations
        self.image_resolution = image_resolution
        self.image_transforms = image_transforms
        
    
    def __len__(self):

        """
        Returns the number of samples in the Dataset.
        """

        return len(self.image_paths)
    

    def __getitem__(self, idx):

        """
        Returns the image and label of specified sample.

        Args:
            idx (int): Index of specified sample.
        """
        
        image = Image.open(self.image_paths[idx]).convert('L')
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.image_transforms:
            image = self.image_transforms(image)

        image = image.repeat(3, 1, 1)
        
        return image, label
            
    def print_dataset_details(self, indices: list = None, subset_name: str = None):

        """
        Prints the class distribution of the Dataset.

        Args:
            indices (list, optional): Specific indices of subset of Dataset to consider.
            subset_name (str, optional): Specific name of subset of Dataset to print.
        """

        if indices is None:
            filtered_labels = self.labels
        else:
            filtered_labels = [self.labels[i] for i in indices]

        filtered_counts = dict(Counter(filtered_labels))

        if subset_name is None:
            print(f'\nTotal Dataset Size: {len(filtered_labels)}')
        else:
            print(f'\n{subset_name} Dataset Size: {len(filtered_labels)}')

        for class_id, class_name in zip(self.class_ids, self.class_names):
            class_prop = filtered_counts[class_id] / len(filtered_labels)

            print(f'Class Name: {class_name} | Class Label: {class_id} | Count: {filtered_counts[class_id]} ' +
                    f'| Prop: {class_prop:.2f}'
                )
            
    def get_dataset_details(self, indices: list = None):

        """
        Get the class distribution of the Dataset.

        Args:
            indices (list, optional): Specific indices of subset of Dataset to consider.
        Output: 
            Dictionary containing: class name, label, count and proportion relative the whole sample    
        """
        
        all_class_names = []
        all_class_labels = []
        all_counts = []
        all_props = []

        if indices is None:
            filtered_labels = self.labels
        else:
            filtered_labels = [self.labels[i] for i in indices]

        filtered_counts = dict(Counter(filtered_labels))

        for class_id, class_name in zip(self.class_ids, self.class_names):
            class_prop = filtered_counts[class_id] / len(filtered_labels)
            
            all_class_names.append(class_name)
            all_class_labels.append(class_id)
            all_counts.append(filtered_counts[class_id])
            all_props.append(class_prop)
            
        data_details ={
            'Class': all_class_names,
            'Label': all_class_labels,
            'Counts': all_counts,
            'Prop': all_props
            } 
            
        return data_details
    
    def split_train_test_val(self, train_prop: float = 0.7, val_prop: float = 0.1, test_prop: float = 0.2, verbose: bool = True):

        """
        Returns indices corresponding to the train, validation and test subsets of the Dataset.

        Args:
            trian_prop (float): Proportion of samples to allocate to the train subset.
            val_prop (float): Proportion of samples to allocate to the validation subset.
            test_prop (float): Proportion of samples to allocate to the test subset.
            verbose (bool): Specifies whether to print distributions of subsets.
        """

        train_split, val_split, test_split = random_split(
            range(len(self)),
            lengths = [train_prop, val_prop, test_prop],
            generator = torch.Generator().manual_seed(self.seed)
        )

        if verbose:
            self.print_dataset_details(train_split.indices, 'Train')
            self.print_dataset_details(val_split.indices, 'Validation')
            self.print_dataset_details(test_split.indices, 'Test')

        return train_split.indices, val_split.indices, test_split.indices
    
    def append_image_transforms(self, image_transforms: transforms.Compose = None, 
                                replace: bool = False, verbose: bool = False):
        
        """
        Appends image transformations to existing transformation pipeline or replaces.
        If multiple `ToTensor()` transformations are included in the resulting pipeline, only the last instance is kept.
        If there are no `ToTensor()` transformations in the resulting pipeline, it is appended.

        Args:
            image_transforms(transfors.Compose, optional): Iterable of image transformations to append.
            replace (bool): Specifies whether to replace with or append the above image_transforms.
            verbose (bool): Specifies whether to print the resulting image transformation pipeline.
        """
        
        if image_transforms is None:
            if self.image_transforms is None:
                image_transforms_list = []
            else: 
                image_transforms_list = self.image_transforms.transforms
        else:
            if replace:
                image_transforms_list = image_transforms.transforms
            else:
                image_transforms_list = self.image_transforms.transforms + image_transforms.transforms

        image_transforms_cleaned = []
        to_tensor_indices = [i for i, tf in enumerate(image_transforms_list) if isinstance(tf, transforms.ToTensor)]

        if to_tensor_indices:
            last_idx = to_tensor_indices[-1]
            image_transforms_cleaned = [tf for i, tf in enumerate(image_transforms_list) if not isinstance(tf, transforms.ToTensor) or i == last_idx]
        else:
            image_transforms_cleaned = image_transforms_list + [transforms.ToTensor()]

        self.image_transforms = transforms.Compose(image_transforms_cleaned)

        if verbose:
            self.print_image_transforms()
            
    def print_image_transforms(self):

        """
        Prints the ordered image transformations applied to the Dataset.
        """

        print('\nCurrent Image Transform Pipeline:')
        for tf in self.image_transforms.transforms:
            print(' ', tf)    
            
    
    def create_dataloaders(self, batch_size: int, train_indices, val_indices, test_indices,
                           image_transforms: transforms.Compose = None, transform_val: bool = False, 
                           train_sample_weights: torch.tensor = None):
        
        """
        Creates the train, validatinon and test DataLoaders required for training a PyTorch model.
        If `train_sample_weights` is specified, they are supplied to WeightedRandomSampler for the train subset.

        Args:
            batch_size (int): Sizes of batches to process samples in DataLoader.
            train_indices (list): Indices corresponding to the train subset of the Dataset.
            val_indices (list): Indices corresponding to the validation subset of the Dataset.
            test_indices (list): Indices corresponding to the test subset of the Dataset.
            image_transforms (transforms.Compose, optional): Additional image transformations for the train subset.
            transform_val (bool): Specifies whether to apply train image transformations to the validation subset.
            train_sample_weights (torch.tensor, optional): Contains weights for each sample in the train subset.
        """

        if image_transforms is not None:
            dataset_aug = copy.deepcopy(self)
            dataset_aug.append_image_transforms(
                image_transforms = image_transforms, verbose = False
            )
            train_dataset = Subset(dataset_aug, train_indices)
            if transform_val:
                val_dataset = Subset(dataset_aug, val_indices)
            else:
                val_dataset = Subset(self, val_indices)
        else:
            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)
        
        test_dataset = Subset(self, test_indices)

        if train_sample_weights is None:
            train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, 
                generator = torch.Generator().manual_seed(self.seed)
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size = batch_size, 
                sampler = WeightedRandomSampler(train_sample_weights, num_samples = len(train_sample_weights), replacement = True)
            )

        val_loader = DataLoader(
            val_dataset, batch_size = batch_size, sampler = SequentialSampler(val_dataset)
        )
        test_loader = DataLoader(
            test_dataset, batch_size = batch_size, sampler = SequentialSampler(test_dataset)
        )

        return train_loader, val_loader, test_loader            
    



from collections import Counter
class HierImageDataset(Dataset):
    
    """
    A custom PyTorch Dataset for creating a dataset with hierarchical labels

    This class handles:
    - Optional image transforms for data augmentation

    Args:
        base_dataset (ImageDataset): The original dataset from which the hierarchical dataset will be built. 
        groups (list): A list of string lists, where each inner list defines how the final nodes are grouped at a given
                       coarser level. The order of the list is important: the first corresponds to the finest
                       grouping, and the last to the broadest. 
        coarse_names (list): A list of string lists, where each inner list defines the names of the groups at a given
                            coarser level. The order of the list is important: the fist corresponds to the finest grouping
                            and the last to the broadest. 
        image_transforms (callable, optional): Image transformations (e.g., data augmentations) to apply. Defaults to None.

        
    Attributes:
        data_directory (str): Path to the dataset root directory (from base_dataset).
        data_subdirectories (list of str): Subdirectories with additional images (from base_dataset).
        seed (int): Random seed used for sampling (from base_dataset).
        class_names (list): List by level with sorted class names included in the dataset.
        class_sizes (torch.Tensor): List by level with the actual sampled size per class.
        class_ids (list): List by level with numeric ID for each class (aligned with `class_names`).
        image_paths (list): List of file paths to all sampled images (from base_dataset).
        labels (list): List by level with numeric class IDs corresponding to each image.
        image_resolution (int): Size to which each image is resized (from base_dataset).
        image_transforms (callable or None): Image transformations applied during training or inference.
        format_file (str): Format of images files. Default: '.tif (from base_dataset)' 
        
    """
       
    def __init__(self, base_dataset,
                #  , groups,coarse_names,
                 adjacency_graph,levels,
                 image_transforms = None,
                 leaves_only=False
        ):
        
        self.data_directory = base_dataset.data_directory
        self.data_subdirectories = base_dataset.data_subdirectories
        self.seed = base_dataset.seed
        self.image_resolution =  base_dataset.image_resolution
        self.image_paths = base_dataset.image_paths
        self.label_to_ids = {name: i for i, name in enumerate(adjacency_graph.keys())}
        self.levels = levels
        
        self.labels = [self.label_to_ids[label] for label in base_dataset.labels]
        self.class_names = []
        self.class_sizes = []
        self.class_ids = []

        # Build hierarchy object
        self.hierarchy = self._build_hierarchy(adjacency_graph, self.label_to_ids)
        # Other class initializations
        self.image_resolution = base_dataset.image_resolution
        self.image_transforms = image_transforms
        if leaves_only:
            self._filter_leaves()
        
    def __len__(self):

        """
        Returns the number of samples in the Dataset.
        """

        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        if self.image_transforms:
            image = self.image_transforms(image)

        image = image.repeat(3, 1, 1)

        label_node = self.labels[idx]

        # path from root to labeled node
        path = self.hierarchy.get_path_to_root(label_node)

        # supervision per node
        targets = {}
        masks = {}

        for parent in path[:-1]:
            children = self.hierarchy.children(parent)

            target_child = path[path.index(parent) + 1]

            targets[parent] = children.index(target_child)

            masks[parent] = [1] * len(children)

        return {
            "image": image,
            "label_node": label_node,
            "path": path,
            "targets": targets,
            "masks": masks
        }
    
    def _build_hierarchy(self, adjacency_graph, label_to_ids):
        """
        Constructs a Hierarchy object from an adjacency graph.
        
        Args:
            adjacency_graph (dict): Dictionary mapping parent -> list of children
            
        Returns:
            Hierarchy: Populated hierarchy object
        """
        hierarchy = Hierarchy()
        
        all_children = set()
        for _, kids in adjacency_graph.items():
            all_children.update(kids)

        # root = node never appearing as child
        root = [n for n in adjacency_graph if n not in all_children][0]

        hierarchy.add_node(label_to_ids[root], root)

        def dfs(parent):
            for child in adjacency_graph[parent]:
                hierarchy.add_node(label_to_ids[child], child, label_to_ids[parent])
                dfs(child)

        dfs(root)
        
        return hierarchy
    

    def _filter_leaves(self):
        """
        Filters image_paths and labels to only include samples
        whose label corresponds to a leaf node in the hierarchy.
        """
        leaf_ids = set(
            node_id for node_id in self.hierarchy.nodes
            if self.hierarchy.is_leaf(node_id)
        )

        filtered_paths, filtered_labels = zip(
            *[(path, label) for path, label in zip(self.image_paths, self.labels)
            if label in leaf_ids]
        ) if self.image_paths else ([], [])

        removed = len(self.image_paths) - len(filtered_paths)

        self.image_paths = list(filtered_paths)
        self.labels = list(filtered_labels)

        print(f"[leaves_only] Kept {len(self.image_paths)} samples | Removed {removed} non-leaf samples")

    def collate_fn(self, batch):
        image = torch.stack([b["image"] for b in batch])
        label_node = [F.one_hot(torch.tensor(b["label_node"], dtype=torch.long), num_classes=len(self.hierarchy)).float() for b in batch]
        label_node = torch.stack(label_node)
        # label_node = [b["label_node"] for b in batch]
        path = [b["path"] for b in batch]
        targets = [b["targets"] for b in batch]
        masks = [b["masks"] for b in batch]
        return {
            "image": image,
            "label_node": label_node,
            "path": path,
            "targets": targets,
            "masks": masks
        }

    def print_dataset_details(self):
        """
        Prints the class distribution of the Dataset.
        """
        print(f'\nTotal Dataset: Size = {len(self)} | Levels = {self.levels}')

        # Get all unique labels (leaf nodes) from the dataset
        leaf_counts = dict(Counter(self.labels))
        
        # Calculate counts for all nodes (including parent nodes)
        all_node_counts = {}
        
        # For each node in the hierarchy, calculate its count
        for node_id in self.hierarchy.nodes.keys():
            if self.hierarchy.is_leaf(node_id):
                all_node_counts[node_id] = leaf_counts.get(node_id, 0)
            else:
                # Parent nodes: sum of all descendant leaf counts
                leaf_descendants = self.hierarchy.descendants(node_id) + [node_id]
                count = 0
                for leaf_id in leaf_descendants:
                    count += leaf_counts.get(leaf_id, 0)
                all_node_counts[node_id] = count

        print(f"all_node_counts: {all_node_counts}\n")

        # Group nodes by their depth in hierarchy
        nodes_by_level = {}
        for node_id in self.hierarchy.nodes.keys():
            depth = self.hierarchy.nodes[node_id].depth
            if depth not in nodes_by_level:
                nodes_by_level[depth] = []
            nodes_by_level[depth].append(node_id)

        print(f"nodes_by_level: {nodes_by_level}\n")

        # Print statistics for each level
        for level in range(self.levels + 1):
            if level not in nodes_by_level:
                continue
            
            level_nodes = sorted(nodes_by_level[level], key=lambda x: self.hierarchy.nodes[x].node_id)
            level_counts = {node_id: all_node_counts[node_id] for node_id in level_nodes}
            
            print(f"\n------------------------Level {level}------------------------")
            
            for node_id in level_nodes:
                node = self.hierarchy.nodes[node_id]
                count = all_node_counts[node_id]
                
                # Skip nodes with zero count
                if count == 0:
                    continue
                
                class_prop = count / len(self)
                node_type = "Leaf" if self.hierarchy.is_leaf(node_id) else "Parent"
                
                print(
                    f'Level: {level} | Class Name: {node.name:20s} | Class Label: {node_id:3d} | '
                    f'Type: {node_type:6s} | Count: {count:6d} | Prop: {class_prop:.2f}'
                )
            
            # Check for missing labels (only relevant for leaf nodes at this level)
            leaf_nodes_at_level = [n for n in level_nodes if self.hierarchy.is_leaf(n)]
            if leaf_nodes_at_level:
                missing_count = sum(1 for label in self.labels 
                                if label is None and self.hierarchy.depth(label) == level)
                if missing_count > 0:
                    print(f'\nImportant: At level {level} there are {missing_count} samples with missing labels\n')
                    
    
    def split_train_test_val(
        self, train_prop: float = 0.7, val_prop: float = 0.1, test_prop: float = 0.2
    ):

        """
        Returns indices corresponding to the train, validation and test subsets of the Dataset.

        Args:
            trian_prop (float): Proportion of samples to allocate to the train subset.
            val_prop (float): Proportion of samples to allocate to the validation subset.
            test_prop (float): Proportion of samples to allocate to the test subset.
        """

        train_split, val_split, test_split = random_split(
            range(len(self)),
            lengths = [train_prop, val_prop, test_prop],
            generator = torch.Generator().manual_seed(self.seed)
        )

        return train_split.indices, val_split.indices, test_split.indices
    
    def append_image_transforms(
        self, image_transforms: transforms.Compose = None, replace: bool = False
        ):        
        """
        Appends image transformations to existing transformation pipeline or replaces.
        If multiple `ToTensor()` transformations are included in the resulting pipeline, only the last instance is kept.
        If there are no `ToTensor()` transformations in the resulting pipeline, it is appended.

        Args:
            image_transforms(transfors.Compose, optional): Iterable of image transformations to append.
            replace (bool): Specifies whether to replace with or append the above image_transforms.
        """
        
        if image_transforms is None:
            if self.image_transforms is None:
                image_transforms_list = []
            else: 
                image_transforms_list = self.image_transforms.transforms
        else:
            if replace:
                image_transforms_list = image_transforms.transforms
            else:
                image_transforms_list = self.image_transforms.transforms + image_transforms.transforms

        image_transforms_cleaned = []
        to_tensor_indices = [i for i, tf in enumerate(image_transforms_list) if isinstance(tf, transforms.ToTensor)]

        if to_tensor_indices:
            last_idx = to_tensor_indices[-1]
            image_transforms_cleaned = [tf for i, tf in enumerate(image_transforms_list) if not isinstance(tf, transforms.ToTensor) or i == last_idx]
        else:
            image_transforms_cleaned = image_transforms_list + [transforms.ToTensor()]

        self.image_transforms = transforms.Compose(image_transforms_cleaned)
                    
    def create_dataloaders(
        self, batch_size: int, train_indices, val_indices, test_indices,
        image_transforms: transforms.Compose = None, transform_val: bool = False, 
        train_sample_weights: torch.tensor = None
    ):
        
        """
        Creates the train, validatinon and test DataLoaders required for training a PyTorch model.
        If `train_sample_weights` is specified, they are supplied to WeightedRandomSampler for the train subset.

        Args:
            batch_size (int): Sizes of batches to process samples in DataLoader.
            train_indices (list): Indices corresponding to the train subset of the Dataset.
            val_indices (list): Indices corresponding to the validation subset of the Dataset.
            test_indices (list): Indices corresponding to the test subset of the Dataset.
            image_transforms (transforms.Compose, optional): Additional image transformations for the train subset.
            transform_val (bool): Specifies whether to apply train image transformations to the validation subset.
            train_sample_weights (torch.tensor, optional): Contains weights for each sample in the train subset.
        """

        if image_transforms is not None:
            dataset_aug = copy.deepcopy(self)
            dataset_aug.append_image_transforms(
                image_transforms = image_transforms, verbose = False
            )
            train_dataset = Subset(dataset_aug, train_indices)
            if transform_val:
                val_dataset = Subset(dataset_aug, val_indices)
            else:
                val_dataset = Subset(self, val_indices)
        else:
            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)
        
        test_dataset = Subset(self, test_indices)

        if train_sample_weights is None:
            train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, 
                collate_fn=self.collate_fn,generator = torch.Generator().manual_seed(self.seed)
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size = batch_size, collate_fn=self.collate_fn,
                sampler = WeightedRandomSampler(train_sample_weights, num_samples = len(train_sample_weights), replacement = True)
            )

        val_loader = DataLoader(
            val_dataset, batch_size = batch_size, sampler = SequentialSampler(val_dataset),
            collate_fn=self.collate_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_size = batch_size, sampler = SequentialSampler(test_dataset),
            collate_fn=self.collate_fn
        )

        return train_loader, val_loader, test_loader   
        
