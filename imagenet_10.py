import json
import os

import torch
from torch.utils.data import dataset, Dataset, Subset
from torchvision import datasets

dataset_location = "../../benchmark_image_datasets/imagenet-1k/train/"

class Imagenet10(Dataset) :
    def __init__(self, imagenet1k_location, sub_to_super_map, class_size=None):
        self.base_ds = datasets.ImageFolder(imagenet1k_location)
        self.sub_to_super_map = sub_to_super_map
        self.subsets = []
        self.supersets = []
        #subImagenet10_class
        
        for sub, super in sub_to_super_map.items():
            self.subsets.append(sub)

            if super not in self.supersets :
                self.supersets.append(super)

    
        full_ds = datasets.ImageFolder(dataset_location)
        self.full_ds = full_ds
        self.idx_to_class = {v: k for k, v in self.full_ds.class_to_idx.items()}

        wanted_label_indices = {
            full_ds.class_to_idx[c]
            for c in self.subsets
        }
    
        subset_indices = [ ]
        subset_indices_labels = [ ]
        for i, (_, y) in enumerate(self.base_ds.samples) :
            if y in wanted_label_indices:
                subset_indices.append(i)
                subset_indices_labels.append(y)


        subset_indices = [
            i for i, (_, y) in enumerate(self.base_ds.samples) 
            if y in wanted_label_indices
        ]

        self.sub_ds = Subset(self.base_ds, subset_indices)

        if class_size != None :
            class_counts = {}
            new_subset_indices = []
            for i, y in enumerate(subset_indices_labels):
                y = self.sub_to_super_map[self.idx_to_class[y]]
                if y not in class_counts:
                    class_counts[y] = 1

                if class_counts[y] > class_size :
                    continue

                new_subset_indices.append(i)

                class_counts[y] += 1

            self.sub_ds = Subset(self.sub_ds, new_subset_indices)

            


    def __len__(self):
        return len(self.sub_ds)


    def __getitem__(self, index):
        return self.sub_ds.__getitem__(index)

    

DS_LOC = "../../benchmark_image_datasets/imagenet-1k/train/"


if __name__ == '__main__':
    fname = "imagenet-10/sub_to_super_map.json"
    with open(fname, "r") as f: sub_to_super = json.load(f)

    ds = Imagenet10(DS_LOC, sub_to_super_map=sub_to_super, class_size=11)
    print(len(ds))


    pass










#import json
#import os

#import torch
#from torch.utils.data import Dataset, Subset
#from torchvision import datasets, transforms


#class Imagenet10(Dataset):
    #"""
    #Wraps an ImageNet-1k ImageFolder and:
      #- keeps only subclasses listed in sub_to_super_map (wnid -> superclass name),
      #- optionally balances to at most `class_size` images per subclass,
      #- returns superclass labels (0..num_superclasses-1) instead of fine-grained labels.
    #"""

    #def __init__(self, imagenet1k_location, sub_to_super_map, class_size=None, transform=None):
        #"""
        #Args:
            #imagenet1k_location (str): Path to ImageNet 'train' directory (folders n0******).
            #sub_to_super_map (dict): {sub_wnid (str): super_name (str), ...}
            #class_size (int or None): Max images per *subclass* (fine class). None = no limit.
            #transform: Optional torchvision transform. If None, ImageFolder's default is used.
        #"""
        ## Base ImageNet dataset
        #self.base_ds = datasets.ImageFolder(imagenet1k_location, transform=transform)
        #self.class_to_idx = self.base_ds.class_to_idx          # wnid -> fine label idx
        #self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}  # fine idx -> wnid

        ## Keep the mapping dict as-is
        #self.sub_to_super_map = sub_to_super_map               # {wnid: super_name}

        ## Collect subclass wnids and superclass names
        #self.sub_wnids = list(self.sub_to_super_map.keys())
        #self.super_names = sorted(set(self.sub_to_super_map.values()))
        #self.super_name_to_idx = {name: i for i, name in enumerate(self.super_names)}

        ## Map fine label index -> superclass label index
        #self.fine_to_super_idx = {}
        #for sub_wnid, super_name in self.sub_to_super_map.items():
            #if sub_wnid not in self.class_to_idx:
                ## This wnid is not present in this ImageNet folder; skip it
                #continue
            #fine_idx = self.class_to_idx[sub_wnid]
            #super_idx = self.super_name_to_idx[super_name]
            #self.fine_to_super_idx[fine_idx] = super_idx

        ## Build list of base dataset indices that belong to any of the selected subclasses
        #subset_indices = []
        #subset_labels_fine = []
        #for i, (_, fine_y) in enumerate(self.base_ds.samples):
            #if fine_y in self.fine_to_super_idx:
                #subset_indices.append(i)
                #subset_labels_fine.append(fine_y)

        ## Optionally limit to at most `class_size` samples per fine-grained subclass
        #if class_size is not None:
            #class_counts = {}
            #filtered_indices = []

            #for base_idx, fine_y in zip(subset_indices, subset_labels_fine):
                #wnid = self.idx_to_class[fine_y]
                #count = class_counts.get(wnid, 0)

                #if count >= class_size:
                    #continue

                #filtered_indices.append(base_idx)
                #class_counts[wnid] = count + 1

            #subset_indices = filtered_indices

        ## Final subset of the base dataset (still returns fine labels)
        #self.sub_ds = Subset(self.base_ds, subset_indices)

    #@property
    #def num_superclasses(self):
        #return len(self.super_names)

    #def __len__(self):
        #return len(self.sub_ds)

    #def __getitem__(self, index):
        ## Get image and fine-grained label from the subset
        #img, fine_y = self.sub_ds[index]

        ## Map fine label -> superclass label
        #super_y = self.fine_to_super_idx[int(fine_y)]

        #return img, super_y


## Example usage
#if __name__ == '__main__':
    #DS_LOC = "../../benchmark_image_datasets/imagenet-1k/train/"
    #fname = "imagenet-10/sub_to_super_map.json"

    #with open(fname, "r") as f:
        #sub_to_super = json.load(f)  # { "n02084071": "dog", "n02121808": "cat", ... }

    ## Optional transform (match what you plan to train with)
    #transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        #transforms.ToTensor(),
    #])

    #ds = Imagenet10(
        #imagenet1k_location=DS_LOC,
        #sub_to_super_map=sub_to_super,
        #class_size=10,        # or None
        #transform=transform,
    #)

    #print("Total samples in Imagenet10:", len(ds))
    #print("Superclasses:", ds.super_names)
    #print("Num superclasses:", ds.num_superclasses)

    ## Quick sanity check: load one sample
    #img, y = ds[0]
    #print("Sample 0 - image shape:", img.shape, "superclass label:", y)
