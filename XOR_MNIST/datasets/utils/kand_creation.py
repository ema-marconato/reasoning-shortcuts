import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np,joblib,glob

from torchvision.datasets.folder import pil_loader

class KAND_Dataset(torch.utils.data.Dataset):
    def __init__(self, base_path, split):
        self.base_path = base_path
        self.split=split
        
        self.list_images= glob.glob(os.path.join(self.base_path,self.split,"images","*"))
        self.img_number = [i for i in range(len(self.list_images))]
        
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.concept_mask=np.array([False]*len(self.list_images))
        self.metas=[]
        self.targets=torch.LongTensor([])
        for item in range(len(self.list_images)):
            target_id=os.path.join(self.base_path,self.split,"meta",str(self.img_number[item]).zfill(3)+".joblib")
            meta=joblib.load(target_id)
            self.metas.append(meta)

    def __getitem__(self, item):
        meta=self.metas[item]
        label=meta["y"]
        # concepts= meta["c"]
        # mask= self.concept_mask[item]
        # if mask:
        #     concepts=-torch.ones_like(concepts)

        concepts = []
        for i in range(5):
            concepts.append( meta['obj'+str(i)]['c'] )
        concepts = np.concatenate(concepts, axis=0)
        # concepts = torch.from_numpy(concepts)

        img_id =  self.img_number[item]
        image_id=os.path.join(self.base_path,self.split,"images",str(img_id).zfill(3)+".png")
        image = pil_loader(image_id)

        return self.transform(image),label,concepts

    def __len__(self):
        return len(self.list_images)

# class OOD_CLEVR(torch.utils.data.Dataset):
#     def __init__(self, base_path):

#         self.base_path = base_path

#         self.list_images= glob.glob(os.path.join(self.base_path,"image","*"))
#         self.task_number = [0] * len(self.list_images)
#         self.img_number = [i for i in range(len(self.list_images))]
#         self.transform = transforms.Compose(
#             [transforms.ToTensor()]
#         )
#         self.concept_mask=np.array([False for i in range(len(self.list_images))])
#         self.metas=[]
#         self.targets=torch.LongTensor([])
#         for item in range(len(self.list_images)):
#             target_id=os.path.join(self.base_path,"meta",str(self.img_number[item])+".joblib")
#             meta=joblib.load(target_id)
#             self.metas.append(meta)

#     @property
#     def images_folder(self):
#         return os.path.join(self.base_path,"image")

#     @property
#     def scenes_path(self):
#         return os.path.join(self.base_path,"image")

#     def __getitem__(self, item):
#         meta=self.metas[item]
#         label=meta["target"]
#         concepts= meta["concepts"]
#         mask= self.concept_mask[item]
#         if mask:
#             concepts=-torch.ones_like(concepts)
#         task_id, img_id = self.task_number[item], self.img_number[item]
#         image_id=os.path.join(self.base_path,"image",str(img_id)+".jpg")
#         image = pil_loader(image_id)
#         return self.transform(image),label,concepts,self.transform(image)

#     def __len__(self):
#         return len(self.list_images)
