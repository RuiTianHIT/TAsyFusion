#
#
# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import torch.nn.functional as f
# import torchvision.transforms.functional as FF
# from torchvision.transforms import InterpolationMode
# from PIL import Image
#
# import pprint
# # import glob
# from numpy.lib.format import open_memmap
# import cv2
# import tqdm
# import matplotlib.pyplot as plt
# import random
# from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop
# import json
# from transformers import CLIPProcessor, CLIPModel
# import time
#
# # --- 1. 初始化模型 (只在程序启动时运行一次) ---
#
# model_name = "openai/clip-vit-large-patch14"
# print(f"正在加载模型 {model_name}...")
# # 加载模型和分词/图像处理器
# clip_model = CLIPModel.from_pretrained(model_name)
# clip_processor = CLIPProcessor.from_pretrained(model_name)
# # device = "cpu"
# device = "cuda"
# clip_model = clip_model.to(device)
# print(f"✅ 模型加载完成，使用设备: {device}")
#
# with open("/home/server/tianrui/describe-anything/train_sequence_00_town01.json", 'r', encoding='utf-8') as f:
#     train_sequence_00_town01 = json.load(f)
# with open("/home/server/tianrui/describe-anything/train_sequence_01_town02.json", 'r', encoding='utf-8') as f:
#     train_sequence_01_town02 = json.load(f)
# with open("/home/server/tianrui/describe-anything/train_sequence_02_town03.json", 'r', encoding='utf-8') as f:
#     train_sequence_02_town03 = json.load(f)
# with open("/home/server/tianrui/describe-anything/train_sequence_03_town04.json", 'r', encoding='utf-8') as f:
#     train_sequence_03_town04 = json.load(f)
# with open("/home/server/tianrui/describe-anything/train_sequence_04_town05.json", 'r', encoding='utf-8') as f:
#     train_sequence_04_town05 = json.load(f)
# with open("/home/server/tianrui/describe-anything/test_sequence_00_town10.json", 'r', encoding='utf-8') as f:
#     test_sequence_00_town10 = json.load(f)
# with open("/home/server/tianrui/describe-anything/valid_sequence_00_town06.json", 'r', encoding='utf-8') as f:
#     valid_sequence_00_town06 = json.load(f)
# with open("/home/server/tianrui/describe-anything/valid_sequence_01_town07.json", 'r', encoding='utf-8') as f:
#     valid_sequence_01_town07 = json.load(f)
#
# class SequenceMVSEC(Dataset):
#     """Load sequences of time-synchronized {event + depth} from a folder."""
#
#     def __init__(self, base_folder, sequence_length=2, transform=None,
#                  step_size=20, clip_distance=100.0, normalize=True):
#
#         assert(sequence_length > 0)
#         assert(step_size > 0)
#         assert(clip_distance > 0)
#         self.L = sequence_length
#         self.dataset = DENSE_Data(base_folder, clip_distance=clip_distance, transform=transform,
#                                       normalize=normalize)
#         self.step_size = step_size
#         if self.L >= len(self.dataset):
#             self.length = 0
#         else:
#             self.length = (len(self.dataset) - self.L) // self.step_size + 1
#         print("lenght sequence dataset: ", self.length)
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, i):
#         """ Returns a list containing synchronized events <-> (event, depth)  pairs
#         """
#         assert(i >= 0)
#         assert(i < self.length)
#         # generate a random seed here, that we will pass to the transform function
#         # of each item, to make sure all the items in the sequence are transformed
#         # in the same way
#         seed = random.randint(0, 2**32)
#         sequence = []
#         # add the first element (i.e. do not start with a pause)
#         k = 0
#         j = i * self.step_size
#         item = self.dataset.__getitem__(j, seed)
#         sequence.append(item)
#         for n in range(self.L - 1):
#             k += 1
#             item = self.dataset.__getitem__(j + k, seed)
#             sequence.append(item)
#         #print(sum(len(k) for k in sequence))
#         return sequence
#
#
# def get_clip_feature(file_name):
#     text=""
#     # print(file_name)
#     # print(file_name.split("/")[4] )
#     if file_name.split("/")[4] == "train_sequence_00_town01":
#         # print("--------------------------", "train_sequence_00_town01")
#         text = train_sequence_00_town01[file_name.split("/")[-1]]
#
#     elif file_name.split("/")[4] == "train_sequence_01_town02":
#         # print("--------------------------", "train_sequence_01_town02")
#         text = train_sequence_01_town02[file_name.split("/")[-1]]
#
#     elif file_name.split("/")[4] == "train_sequence_02_town03":
#         # print("--------------------------", "train_sequence_02_town03")
#         text = train_sequence_02_town03[file_name.split("/")[-1]]
#
#     elif file_name.split("/")[4] == "train_sequence_03_town04":
#         # print("--------------------------", "train_sequence_03_town04")
#         text = train_sequence_03_town04[file_name.split("/")[-1]]
#
#     elif file_name.split("/")[4] == "train_sequence_04_town05":
#         # print("--------------------------", "train_sequence_04_town05")
#         text = train_sequence_04_town05[file_name.split("/")[-1]]
#
#     elif file_name.split("/")[4] == "test_sequence_00_town10":
#
#         text = test_sequence_00_town10[file_name.split("/")[-1]]
#
#     elif file_name.split("/")[4] == "valid_sequence_00_town06":
#
#         text = valid_sequence_00_town06[file_name.split("/")[-1]]
#
#     elif file_name.split("/")[4] == "valid_sequence_01_town07":
#
#         text = valid_sequence_01_town07[file_name.split("/")[-1]]
#
#
#
#     # print(text)
#     start_time = time.perf_counter()
#     inputs = clip_processor(
#         text=text,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=77,
#         return_attention_mask=True)
#     inputs = {k: v.to(device) for k, v, in inputs.items()}
#     with torch.no_grad():
#         text_outputs = clip_model.text_model(
#             input_ids=inputs['input_ids'],
#             attention_mask=inputs['attention_mask'])
#         text_features_768 = text_outputs.last_hidden_state
#     text_features_768 = text_features_768 / text_features_768.norm(p=2, dim=-1, keepdim=True)
#     text_features_768 = text_features_768[:, 0, :]
#     end_time = time.perf_counter()
#     elapsed_time_ms = (end_time - start_time) * 1000
#
#     print(f"代码运行耗时************************////////////////////////: {elapsed_time_ms:.2f} 毫秒")
#     # print("tiaoshi1:", text_features_768.shape)
#     return text_features_768
#
#
# class DENSE_Data(Dataset):
#     def __init__(self, base_folder,
#                  clip_distance=80.0,
#                  transform=None,
#                  normalize=True):
#
#         self.base_folder = base_folder
#         self.transform = transform
#         self.normalize = normalize
#         self.eps = 1e-06
#         self.clip_distance = clip_distance
#
#         self.fill_out_nans = False
#
#         files = os.listdir(self.base_folder + '/events/frames_white')
#
#         self.files_list = list(filter(self.file_filter, files))
#
#
#
#     def __len__(self):
#         l = 0
#         # files = os.listdir(self.base_folder + '/event_voxel')
#         l += len(self.files_list)
#         return l
#
#     def file_filter(self, file):
#         if file[-4:] in ['.jpg', '.png', '.bmp']:
#             return True
#         else:
#             return False
#
#     def prepare_depth(self, depth, seed, reg_factor):
#         # Clip to maximum distance
#         depth = np.clip(depth, 0.0, self.clip_distance)
#         # Normalize
#         depth = depth / self.clip_distance
#         # Convert to log depth
#         depth = 1.0 + np.log(depth + self.eps) / reg_factor
#         # Clip between 0 and 1.0
#         depth = depth.clip(0, 1.0)
#
#         if len(depth.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
#             depth = np.expand_dims(depth, -1)
#
#         depth = np.moveaxis(depth, -1, 0)  # H x W x C -> C x H x W
#
#         if self.fill_out_nans:
#             upper, lower = np.vsplit(depth[0], 2)
#             upper = np.nan_to_num(upper, nan=1.0)
#             # lower = np.nan_to_num(lower, nan=0.0)
#             depth = np.vstack([upper, lower])
#             depth = depth[None, :]
#
#         depth = torch.from_numpy(depth)  # numpy to tensor
#
#         if self.transform:
#             random.seed(seed)
#             depth = self.transform(depth)
#         return depth
#
#     def rgb2gray(self, rgb):
#         return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
#
#     def prepare_frame(self, frame, seed):
#         if len(frame.shape) > 2:
#             if frame.shape[2] > 1:
#                 frame = self.rgb2gray(frame)  # [H x W]
#
#         frame /= 255.0  # normalize
#         frame = np.expand_dims(frame, axis=0)  # expand to [1 x H x W]
#         frame = torch.from_numpy(frame)
#         if self.transform:
#             random.seed(seed)
#             frame = self.transform(frame)
#         return frame
#
#     def normalize_voxelgrid(self, event_tensor):
#
#         mask = np.nonzero(event_tensor)
#         if mask[0].size > 0:
#             mean, stddev = event_tensor[mask].mean(), event_tensor[mask].std()
#             if stddev > 0:
#                 event_tensor[mask] = (event_tensor[mask] - mean) / stddev
#         return event_tensor
#
#     def __getitem__(self, i, seed=None, reg_factor=3.70378):
#         """
#                 This function returns a tuple (event_voxel, depth, frame) where depth is the label
#
#                 """
#
#         # item = []
#         assert(i >= 0)
#
#         if seed is None:
#             # if no specific random seed was passed, generate our own.
#             # otherwise, use the seed that was passed to us
#             seed = random.randint(0, 2 ** 32)
#
#         self.files_list.sort(key=lambda x: int(x[7:-4]))
#
#         index = self.files_list[i].split('.')[0][7:]
#
#         event_path = self.base_folder + '/events/frames_white/events_' + index + '.png'
#
#         event_color = cv2.imread(event_path).astype(np.float32)
#
#         image_path = self.base_folder + '/rgb/frames/frame_' + index + '.png'
#         image_color = cv2.imread(image_path).astype(np.float32)  # load rgb image info
#
#         # Random scaling
#         if self.transform:
#             scale = [0.9, 1, 1.1]
#             random.seed(seed)
#             ratio = random.choice(scale)
#             hh = int(ratio * 260)
#             ww = int(ratio * 346)
#             event_color = Image.fromarray(np.uint8(event_color))
#             image_color = Image.fromarray(np.uint8(image_color))
#
#             if ratio <= 1:
#                 event_color = FF.resize(event_color, (hh, ww), interpolation=InterpolationMode.NEAREST)
#                 image_color = FF.resize(image_color, (hh, ww), interpolation=InterpolationMode.NEAREST)
#             else:
#                 event_color = FF.resize(event_color, (hh, ww), interpolation=InterpolationMode.BICUBIC)
#                 image_color = FF.resize(image_color, (hh, ww), interpolation=InterpolationMode.BICUBIC)
#
#             event_color = np.array(event_color).astype(np.float32)
#             image_color = np.array(image_color).astype(np.float32)
#
#         event_color /= 255.0  # normalize
#         event_color = np.moveaxis(event_color, -1, 0) # H x W x C -> C x H x W
#         event_color = torch.from_numpy(event_color)
#
#
#
#         image_color /= 255.0  # normalize
#         image_color = np.moveaxis(image_color, -1, 0)
#         image_color = torch.from_numpy(image_color)
#
#
#         if self.transform:
#             random.seed(seed)
#             event_color = self.transform(event_color)
#             image_color = self.transform(image_color)
#         else:
#             event_color = event_color[:, 33:257, 121:345]
#             image_color = image_color[:, 33:257, 121:345]
#             # event_color = event_color[:, 1:257, 5:341]
#
#
#         depth_path = self.base_folder + '/depth/data/depth_' + index + '.npy'
#         depth = np.load(depth_path).astype(np.float32)
#         if self.transform:
#             scale = [0.9, 1, 1.1]
#             random.seed(seed)
#             ratio = random.choice(scale)
#             hh = int(ratio * 260)
#             ww = int(ratio * 346)
#             depth = Image.fromarray(depth)
#
#             if ratio <= 1:
#                 depth = FF.resize(depth, (hh,ww), interpolation=InterpolationMode.NEAREST)
#             else:
#                 depth = FF.resize(depth, (hh,ww), interpolation=InterpolationMode.BICUBIC)
#
#             depth = np.array(depth)
#             depth = depth / ratio
#         else:
#             # depth = depth[1:257, 5:341]
#             depth = depth[33:257, 121:345]
#         depth = self.prepare_depth(depth, seed, reg_factor)
#         # item += [{"depth": depth}]
#
#         text_features = get_clip_feature(image_path)
#
#
#         item = {'event': event_color,
#                 'depth': depth,
#                 'rgb': image_color,
#                 'text': text_features}
#
#         return item
#
# def collate_mvsec(batch):
#     # batch: list of length batch_size. Each entry is a sequence of dicts, that contains the data
#     # (keys: imageX, eventsX, depth, image_loss). size of batch: batch_size, seq_length, dict_entries, [1, x, x, x]
#     # return_sequence should be a sequence of dicts, where each dict contains all corresponding
#     # entries of the whole batch. size of return sequence: seq_length, dict entries, [batch_size, x, x, x]
#     return_sequence = []
#     sequence_length = len(batch[0])
#     batch_size = len(batch)
#     for j in range(sequence_length):
#         # loop over the whole sequence to fill return_sequence list
#         return_sequence += [[batch[i][j] for i in range(batch_size)]]
#     return return_sequence
#
#
#
#
