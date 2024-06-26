"""

Args : 
    - number of folds (in this problem, we decide 5-fold as k-fold validation)
    - index of the fold
    
Returns : 
    - datasets/kaist_rgbt/train_{idx}_{fold}.txt
    - datasets/kaist_rgbt/val_{idx}_{fold}.txt 
    - datasets/KAIST_annotation.json

"""

import os
import json
import math 
import argparse

clss_map = {
    "person": 0,
    "cyclist": 1,
    "people": 2,
    "person?": 3
}

class K_fold_validation():
    def __init__(self, args):
        self.dataset = args.dataset 
        self.root_path = os.environ.get("PWD") + "/" + self.dataset

        self.train_txt = self.root_path + "/" + "train-all-04.txt"
        with open(self.train_txt, 'r') as f:
            self.file_list = f.readlines()

        # Options for K-cross validation
        self.k_cross_val = args.k
        self.k_index = args.idx 
        self.imgs_per_fold = math.floor(len(self.file_list) / self.k_cross_val)

        # Files
        self.train_list = []
        self.train_txt_file = self.root_path + "/" + f"train_{self.k_index}_{self.k_cross_val}.txt"
        self.val_list = []
        self.val_txt_file = self.root_path + "/" + f"val_{self.k_index}_{self.k_cross_val}.txt"
        
        self.annotation_file = self.root_path + "/annotations/" + f"annot_{self.k_index}_{self.k_cross_val}.json"
        self.annotation_file_example = self.root_path + "/annotations/example_annot.json"
        self.annots = None
        self.annot_imgs = []
        self.annot_annots = []
        self.annot_categories = {}

        assert self.k_index < 5 and self.k_index >=0, "k_index shold be greater than 0 and less than 5"
        for line, file_name in enumerate(self.file_list):
            if line >= self.imgs_per_fold * self.k_index and line <= self.imgs_per_fold * (self.k_index + 1):
                self.val_list.append(file_name)
            else:
                self.train_list.append(file_name)        

    def gen_train(self):
        with open(self.train_txt_file, "w+") as f:
            for line in self.train_list:
                f.write(line)

    def gen_val(self):
        with open(self.val_txt_file, "w+") as f:
            for line in self.val_list:
                f.write(line)

    def gen_annot(self):
        
        with open(self.annotation_file_example, "r") as example:
            baseline = json.load(example)
        self.annot_categories = baseline["categories"]

        # TODO : images
        for idx, line in enumerate(self.val_list):
            
            tmp_imgs = {
                "id": idx,
                "im_name": line.split("\n")[0],
                "height": 512,
                "width": 640
            }
            self.annot_imgs.append(tmp_imgs)

        # TODO : annotations

        ## read xml file 
        import xml.etree.ElementTree as ET
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        self.xml_path = self.root_path + "/train/labels-xml/"
        self.id = 0 
        for idx, line in enumerate(self.val_list):
            file_name = line.split("/")[-1].split("\n")[0]
            self.xml_file = file_name.replace(".jpg",".xml")
            
            with open(self.xml_path+self.xml_file) as f :
                tree = ET.parse(f)
                root = tree.getroot()

            for data in root:
                bbox = []
                if data.tag == "object":
                    for element in data.getchildren():
                        if element.tag == "name":
                            name = element.text
                        
                        if element.tag == "bndbox":
                            for xywh in element.getchildren():
                                bbox.append(int(xywh.text))
                        if element.tag == "occlusion":
                            occlusion = int(element.text)
                            
                    tmp_annots = {
                        "id": self.id,
                        "image_id": idx,
                        "category_id": clss_map[name],
                        "bbox": bbox,
                        "height": bbox[-1],
                        "occlusion": occlusion,
                        "ignore": 0
                    }
                    self.id += 1
                    self.annot_annots.append(tmp_annots)
    
    # TODO : merge dictionarys to one json file 
    def merge_keys(self):
        self.annots = {
            "images": self.annot_imgs,
            "annotations": self.annot_annots,
            "categories": self.annot_categories
        }

        self.json_name = f"KAIST_annotation_{self.k_index}_{self.k_cross_val}.json" 
        with open(self.json_name, "w") as f:
            json.dump(self.annots, f)

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--k", type = int, default = 5)
    args.add_argument("--idx", type = int, default = 2)
    args.add_argument("--dataset", type = str, default = "kaist-rgbt")
    opt = args.parse_args()
    
    k_fold_val = K_fold_validation(opt)
    
    k_fold_val.gen_train()
    k_fold_val.gen_val()
    k_fold_val.gen_annot()

    k_fold_val.merge_keys()
    
    print("Done!")
