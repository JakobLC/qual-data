
import numpy as np
from PIL import Image
from pathlib import Path
import os
import cv2
import copy
import json, jsonlines
import tqdm

def downscale_to_atmost_pixels(image,max_num_pixels=1e6,interpolation=3):
    num_pixels = image.shape[0]*image.shape[1]
    if num_pixels>max_num_pixels:
        scale = np.sqrt(max_num_pixels/num_pixels)
        new_shape = (int(np.floor(image.shape[0]*scale)),int(np.floor(image.shape[1]*scale)))
        image = cv2.resize(image,new_shape,interpolation=interpolation) #3=cv2.INTER_AREA
    return image

def save_dict_list_to_json(data_list, file_path, append=False):
    assert isinstance(file_path,str), "file_path must be a string"
    if not isinstance(data_list,list):
        data_list = [data_list]
    
    if file_path.endswith(".json"):
        loaded_data = []
        if append:
            if Path(file_path).exists():
                loaded_data = load_json_to_dict_list(file_path)
                if not isinstance(loaded_data,list):
                    loaded_data = [loaded_data]
        data_list = loaded_data + data_list
        with open(file_path, "w") as json_file:
            json.dump(data_list, json_file, indent=4)
    else:
        assert file_path.endswith(".jsonl"), "File path must end with .json or .jsonl"
        mode = "a" if append else "w"
        with jsonlines.open(file_path, mode=mode) as writer:
            for line in data_list:
                writer.write(line)

def load_json_to_dict_list(file_path):
    assert len(file_path)>=5, "File path must end with .json"
    assert file_path[-5:] in ["jsonl",".json"], "File path must end with .json or .jsonl"
    if file_path[-5:] == "jsonl":
        assert len(file_path)>=6, "File path must end with .json or .jsonl"
        assert file_path[-6:]==".jsonl","File path must end with .json or .jsonl"
    if file_path[-5:] == ".json":
        with open(file_path, 'r') as json_file:
            data_list = json.load(json_file)
    elif file_path[-6:] == ".jsonl":
        data_list = []
        with jsonlines.open(file_path) as reader:
            for line in reader:
                data_list.append(line)
    return data_list

def open_image(image_path,num_channels=None,make_0_1_float=False):
    assert image_path.find(".")>=0, "image_path must contain a file extension"
    image = np.array(Image.open(image_path))
    if num_channels is not None:
        assert num_channels in [0,1,3,4], f"Expected num_channels to be in [0,1,3,4], got {num_channels}"
        if num_channels==0: #means only 2 dims
            if (len(image.shape)==3 and image.shape[2]==1):
                image = image[:,:,0]
            else:
                assert len(image.shape)==2, f"loaded image must either be 2D or have 1 channel when num_channels=0. got shape: {image.shape}"
        else:
            if len(image.shape)==2:
                image = image[:,:,None]
            if num_channels==1:
                assert image.shape[2]==1, f"loaded image must have at most 1 channel if num_channels==1, found {image.shape[2]}"
            elif num_channels==3:
                if image.shape[2]==1:
                    image = np.repeat(image,num_channels,axis=-1)
                elif image.shape[2]==4:
                    image = image[:,:,:3]
                else:
                    assert image.shape[2]==3, f"loaded image must have 1,3 or 4 channels if num_channels==3, found {image.shape[2]}"
            elif num_channels==4:
                if image.shape[2]==1:
                    image = np.concatenate([image,image,image,np.ones_like(image)*255],axis=-1)
                elif image.shape[2]==3:
                    image = np.concatenate([image,np.ones_like(image[:,:,0:1])*255],axis=-1)
                else:
                    assert image.shape[2]==4, f"loaded image must have 1,3 or 4 channels if num_channels==4, found {image.shape[2]}"
    if make_0_1_float:
        image = image.astype(float)/255
    else:
        pass
    return image

def str_to_seed(s):
    return int("".join([str(ord(l)) for l in s]))%2**32

def get_named_datasets(datasets,datasets_info=None,data_root="./data/"):
    if datasets_info is None:
        datasets_info = load_json_to_dict_list(str(Path(data_root)/ "datasets_info_live.json"))
    
    if not isinstance(datasets,list):
        assert isinstance(datasets,str), "expected datasets to be a string or a list of strings"
        datasets = datasets.split(",")
    named_datasets_criterion = {"non-medical": lambda d: d["type"]=="pictures",
                                    "medical": lambda d: d["type"]=="medical",
                                    "all": lambda d: True,
                                    "high-qual": lambda d: d["quality"]=="high",
                                    "non-low-qual": lambda d: d["quality"]!="low",
                                    "binary": lambda d: d["num_classes"]==2,}
    dataset_list = copy.deepcopy(datasets)
    if len(datasets)==1:
        if datasets[0] in named_datasets_criterion:
            crit = named_datasets_criterion[datasets[0]]
            dataset_list = [d["dataset_name"] for d in datasets_info if d["live"] and crit(d)]

    available_datasets = [d["dataset_name"] for d in datasets_info if d["live"]]
    assert all([d in available_datasets for d in dataset_list]), "Unrecognized dataset. Available datasets are: "+str(available_datasets)+". Named groups of datasets are: "+str(list(named_datasets_criterion.keys()))+" got "+str(dataset_list)
    
    return dataset_list

def sam_resize_index(h,w,resize=64):
    if h>w:
        new_h = resize
        new_w = np.round(w/h*resize).astype(int)
    else:
        new_w = resize
        new_h = np.round(h/w*resize).astype(int)
    return new_h,new_w

def pad_to_square(image,pad_value=0):
    h,w = image.shape[:2]
    if h==w:
        return image
    elif h>w:
        pad_right = h-w
        return np.pad(image,(0,pad_right,(0,0)),constant_values=pad_value)
    else:
        pad_bottom = w-h
        return np.pad(image,((0,pad_bottom),(0,0),(0,0)),constant_values=pad_value)

class NonTorchSegmentationDataset():
    def __init__(self,
                      image_reshape_size=None,
                      pad_to_square=False,
                      datasets="all",
                      info_str="info_subset",
                      shuffle_classes=False,
                      data_root="./data/",
                      seg_as_uint8=False,
                      image_as_uint8=False,
                      map_excess_classes_to="modulus",
                      resize_method="area",
                      padding_idx=255):
        """
        The NonTorchSegmentationDataset class is a class to load 
        data from many different datasets. It has useful 
        processing functionalities.
        Inputs:
            image_reshape_size: int or None, the size to reshape the 
                images to. If None, the images are not reshaped but 
                kept as differently shaped items in a list when
                loaded. The longest side of the image is resized to
                this size.

            pad_to_square: bool, if True, the images are padded to
                be square. If False, the images are padded to be

            datasets: str or list of str, the datasets to load. If
                "all", all datasets are loaded. If a list, or  a
                comma seperated string, then only datasets in the 
                list are loaded.

            shuffle_classes: bool, if True, then classes of loaded
                images are shuffled (mainly to vary the colors of
                visualizations)

            data_root: str, the root directory of the data folder

            seg_as_uint8: bool, if True, then the segmentations
                are loaded as uint8. If False, the data is loaded as 
                int

            image_as_uint8: bool, if True, then the images are loaded
                as uint8. If False, the data is loaded as float in the
                range [0,1]

            map_excess_classes_to: str, one of ["largest",
                "random_different","modulus","zero"] which determines
                how to map excess classes above 255 if seg_as_uint8 is
                True. "largest" maps all excess classes to the largest.
                "random_different" maps all excess classes to a random
                class. "modulus" maps all excess classes to the class 
                modulo 256. "zero" maps all excess classes to zero.

            resize_method: str, the method to use for resizing images.
                One of ["nearest","linear","area","cubic","lanczos4"]

            padding_idx: int, the index of the padding class in the
                label maps. If data_as_uint8 is True, then the 
                padding index must be a uint8 value.
        """
        self.datasets = datasets

        self.pad_to_square = pad_to_square
        self.image_reshape_size = image_reshape_size
        self.shuffle_classes = shuffle_classes
        self.data_root = data_root
        self.seg_as_uint8 = seg_as_uint8
        self.image_as_uint8 = image_as_uint8
        self.padding_idx = padding_idx
        
        resize_method_dict = {"nearest": cv2.INTER_NEAREST,
                               "linear": cv2.INTER_LINEAR,
                                 "area": cv2.INTER_AREA,
                                "cubic": cv2.INTER_CUBIC,
                             "lanczos4": cv2.INTER_LANCZOS4}
        assert resize_method in resize_method_dict, "invalid resize_method. Must be one of "+str(list(resize_method_dict.keys()))+", got "+resize_method
        self.resize_method = resize_method_dict[resize_method]

        
        if self.image_reshape_size is not None:
            assert isinstance(self.image_reshape_size,int), "expected image_reshape_size to be an integer"
            assert self.image_reshape_size>0, "expected image_reshape_size to be positive"
        
        assert map_excess_classes_to in ["largest","random_different","modulus","zero"]
        self.map_excess_classes_to = map_excess_classes_to

        self.datasets_info = load_json_to_dict_list(str(Path(data_root) / "datasets_info_live.json"))

        
        self.dataset_list = get_named_datasets(self.datasets,datasets_info=self.datasets_info)
        
        self.items = []
        self.length = 0
        self.idx_to_class = {}
        self.didx_to_item_idx = []
        self.datasets_info = {d["dataset_name"]: d for d in self.datasets_info if d["dataset_name"] in self.dataset_list}

        for dataset_name in tqdm.tqdm(self.dataset_list):
            info_json = load_json_to_dict_list(os.path.join(self.data_root,dataset_name,info_str+".jsonl"))
            N = len(info_json)
            items = []
            if len(items)==0:
                file_format = self.datasets_info[dataset_name]["file_format"]
                for item in info_json:
                    idx = item["i"]
                    item["image_path"] = os.path.join("files",str(idx)+"_im."+file_format)
                    item["label_path"] = os.path.join("files",str(idx)+"_la.png")
                    item["dataset_name"] = dataset_name
                    items.append(item)
            class_dict = load_json_to_dict_list(os.path.join(self.data_root,dataset_name,"idx_to_class.json"))[0]
            self.idx_to_class[dataset_name] = class_dict
            assert len(class_dict)==self.datasets_info[dataset_name]["num_classes"], ("num_classes in idx_to_class.json does not match num_classes in info found "+
                                                                                      str(len(class_dict))+" and "+str(self.datasets_info[dataset_name]["num_classes"])+
                                                                                      " for dataset "+dataset_name)
            self.didx_to_item_idx.extend([f"{dataset_name}/{i}" for i in range(N)])
            self.length += len(items)
            self.items.extend(items)

        self.didx_to_item_idx = {k: i for i,k in enumerate(self.didx_to_item_idx)}
        self.len_per_dataset = {dataset_name: len([item for item in self.items if item["dataset_name"]==dataset_name]) for dataset_name in self.dataset_list}
        
    def __len__(self):
        return self.length
    
    def convert_to_idx(self,list_of_things):
        """
        Converts a list of things to a list of indices. The items in
        the list should either be:
         - a list of integer indices (where we only check that the indices are valid) e.g. 123
         - a list of info dicts with the fields "dataset_name" and "i". e.g. {"dataset_name": "cityscapes","i": 216}
         - a list of strings formatted like '{dataset_name}/{i}', e.g. 'cityscapes/216'
        Returns a list of integer indices and checks they are valid
        """
        assert isinstance(list_of_things,list)
        if len(list_of_things)==0: return []
            
        item0 = list_of_things[0]
        if isinstance(item0,int):
            list_of_things2 = list_of_things
        elif isinstance(item0,dict):
            assert "dataset_name" in item0 and "i" in item0, "item0 must be a dict with the fields 'dataset_name' and 'i'"
            d_vec = [item["dataset_name"] for item in list_of_things]
            i_vec = [item["i"] for item in list_of_things]
        elif isinstance(item0,str):
            d_vec = [item.split("/")[0] for item in list_of_things]
            i_vec = [int(item.split("/")[1]) for item in list_of_things]
        else:
            raise ValueError(f"Unrecognized type for item0: {type(item0)}, should be int, dict or str")

        if isinstance(item0,(dict,str)):
            list_of_things2 = []
            for d,i in zip(d_vec,i_vec):
                match_idx = None
                for k,item in enumerate(self.items):
                    if item["dataset_name"]==d and item["i"]==i:
                        match_idx = k
                        break   
                assert match_idx is not None, "No match for dataset_name: "+d+", i: "+str(i)
                list_of_things2.append(match_idx)

        assert all([isinstance(item,int) for item in list_of_things2]), "all items in list_of_things must be integers"
        assert all([0<=item<len(self) for item in list_of_things2]), "all items in list_of_things must be valid indices"
        return list_of_things2

    def preprocess(self,image,label,info):
        if self.image_reshape_size is None:
            if self.pad_to_square:
                pad = pad_to_square
                pad_label = lambda x: pad_to_square(x,pad_value=self.padding_idx)
                resize = lambda x: x
            else:
                pad = lambda x: x
                pad_label = lambda x: x
                resize = lambda x: x
        else:
            if self.pad_to_square:
                pad = lambda x: pad_to_square
                pad_label = lambda x: pad_to_square(x,pad_value=self.padding_idx)
                resize = lambda x: cv2.resize(x,(self.image_reshape_size,self.image_reshape_size),interpolation=self.resize_method)
            else:
                pad = lambda x: x
                pad_label = lambda x: x
                h,w = image.shape[:2]
                new_h,new_w = sam_resize_index(h,w,self.image_reshape_size)
                resize = lambda x: cv2.resize(x,(new_w,new_h),interpolation=self.resize_method)
        image = resize(pad(image))
        label = resize(pad_label(label))
        if self.shuffle_classes:
            perm = np.random.permutation(len(info["classes"]))
            label = np.vectorize(lambda x: perm[x])(label)
            #edit "classes" and "class_counts" to reflect the new order
            info["classes"] = [info["classes"][i] for i in perm]
            info["class_counts"] = [info["class_counts"][i] for i in perm]
        itc_d = self.idx_to_class[info["dataset_name"]]
        idx_to_class_for_image = {i: itc_d[str(idx)] for i,idx in enumerate(info["classes"])}
        info["idx_to_class"] = idx_to_class_for_image
        return image,label,info

    def __getitem__(self, idx):
        info = copy.deepcopy(self.items[idx]) #copy to avoid changing the original
        dn = info["dataset_name"]
        image = open_image(os.path.join(self.data_root,dn,info["image_path"]),num_channels=3,make_0_1_float=True)
        label = open_image(os.path.join(self.data_root,dn,info["label_path"]),num_channels=0) #0 channels is 2d
        image,label,info = self.preprocess(image,label,info)
        return image,label,info

    def get_random_batch(self,batch_size=8,limited_to_dataset=None,downscale_num_pixels=None):
        if limited_to_dataset is None:
            limited_to_dataset = self.dataset_list
        else:
            if isinstance(limited_to_dataset,str):
                if limited_to_dataset.find(",")>=0:
                    limited_to_dataset = limited_to_dataset.split(",")
            assert all([d in self.dataset_list for d in limited_to_dataset]), "Unrecognized dataset. Available datasets are: "+str(self.dataset_list)+" got "+str(limited_to_dataset)
        
        valid_idxs = [i for i in range(len(self)) if self.items[i]["dataset_name"] in limited_to_dataset]
        assert len(valid_idxs)>=batch_size, "not enough valid indices to sample from"
        batch_indices = np.random.choice(valid_idxs,batch_size,replace=False)
        images = []
        labels = []
        infos = []
        for idx in batch_indices:
            image,label,info = self[idx]
            if downscale_num_pixels:
                image = downscale_to_atmost_pixels(image,max_num_pixels=downscale_num_pixels,interpolation=cv2.INTER_AREA)
                label = downscale_to_atmost_pixels(label,max_num_pixels=downscale_num_pixels,interpolation=cv2.INTER_NEAREST)
            images.append(image)
            labels.append(label)
            infos.append(info)
        return images,labels,infos

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", type=int, default=-1)
    args = parser.parse_args()
    if args.process==0:
        print("Process number 0: create dataset and load the first item")
        ntsd = NonTorchSegmentationDataset()
        item0 = ntsd[0]
        print("item0 keys: "+str(item0.keys()))
        print("len(ntsd): "+str(len(ntsd)))
    else:
        raise ValueError("Invalid process number: "+str(args.process))
        
if __name__=="__main__":
    main()