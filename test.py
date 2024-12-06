# /coc/testnvme/chuang475/projects/VQA-ICL/cache/rices/test/ok_vqa.pkl
# /coc/testnvme/chuang475/projects/VQA-ICL/cache_bs1/rices/test/ok_vqa.pkl

# check if the two files are the same
import os
import torch

file1 = "/coc/testnvme/chuang475/projects/VQA-ICL/cache/rices/train/ok_vqa.pkl"
file2 = "/coc/testnvme/chuang475/projects/VQA-ICL/cache_bs1/rices/train/ok_vqa.pkl"

data1 = torch.load(file1, map_location="cpu")
data2 = torch.load(file2, map_location="cpu")

print(data1.shape)
print(data2.shape)

# check if the two files are the same within a tolerance
print(torch.allclose(data1, data2, atol=1e-2))

# print out the different elements of data1[-1] and data2[-1]
for j in range(data1.shape[0]):
    if torch.allclose(data1[j], data2[j], atol=1e-4) == False:
        # print(j, data1[j], data2[j])
        # print(j)
        break

for i in range(data1[-1].shape[0]):
    if torch.allclose(data1[-1][i], data2[-1][i], atol=1e-4) == False:
        print(i, data1[-1][i], data2[-1][i])
