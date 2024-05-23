import torch
import numpy as np
from torch import nn
import torchvision
import model


LOAD_PATH = "path_to_file.pth"
device = torch.device('cpu')

scale_factor = 12
DCE_net = model.simpleEnhanceNet(scale_factor).to(device)
DCE_net.load_state_dict(torch.load(LOAD_PATH))

# print("Structure : \n", DCE_net)
# print("=" * 80)
# print("Exporting weight file " + LOAD_PATH)
# LOAD_PATH = LOAD_PATH.replace('pth', 'bin')
# with open(LOAD_PATH, "wb") as file:
#     for param_name in DCE_net.state_dict():
#         if param_name.find("num_batches_tracked") != -1:
#             continue
#         layer_weight = DCE_net.state_dict()[param_name].flatten().numpy()
#         for weight in layer_weight:
#             file.write(weight)
# print(LOAD_PATH + " Weight file exported")
# exit()

weight_list = list(DCE_net.state_dict().keys())
print("weigth_list : ", weight_list)



# for i in weight_list:
#     w = DCE_net.state_dict()[i].to(device).numpy()
#     w = w.flatten()
#     if i.find(".weight") != -1:
#         print(i, "\t", w.shape, "\t", (w.max() * 16384).astype('int16'), (w.min() * 16384).astype('int16'))
#     elif i.find(".bias") != -1:
#         print(i, "\t", w.shape, "\t", (w.max() * 16384 * 16384).astype('int32'), (w.min() * 16384 * 16384).astype('int32')) # 2^28 => 268435456

# exit()

with open("qZeroDCE_Weight.h", "w") as f:
    print("export ", weight_list[0])
    f.write("const short conv1_w[864] = {\n")
    w = DCE_net.state_dict()[weight_list[0]].to(device).numpy()
    w = w.flatten()
    count = 0
    for param in w:
        count += 1
        f.write(str(int(param * 16384)))

        if count != w.size:
            f.write(", ")
        else:
            f.write("};\n")

        if count % 16 == 0:
            f.write("\n")
    f.write("\n")

    print("export ", weight_list[1])
    f.write("const int conv1_b[32] = {\n")
    w = DCE_net.state_dict()[weight_list[1]].to(device).numpy()
    w = w.flatten()
    count = 0
    for param in w:
        count += 1
        f.write(str(int(param * 16384 * 16384)))

        if count != w.size:
            f.write(", ")
        else:
            f.write("};\n")

        if count % 8 == 0:
            f.write("\n")
    f.write("\n")

    print("export ", weight_list[2])
    f.write("const short conv2_w[9216] = {\n")
    w = DCE_net.state_dict()[weight_list[2]].to(device).numpy()
    w = w.flatten()
    count = 0
    for param in w:
        count += 1
        f.write(str(int(param * 16384)))

        if count != w.size:
            f.write(", ")
        else:
            f.write("};\n")

        if count % 16 == 0:
            f.write("\n")
    f.write("\n")

    print("export ", weight_list[3])
    f.write("const short conv2_b[32] = {\n")
    w = DCE_net.state_dict()[weight_list[3]].to(device).numpy()
    w = w.flatten()
    count = 0
    for param in w:
        count += 1
        f.write(str(int(param * 16384 * 16384)))

        if count != w.size:
            f.write(", ")
        else:
            f.write("};\n")

        if count % 8 == 0:
            f.write("\n")
    f.write("\n")

    print("export ", weight_list[4])
    f.write("const short conv3_w[864] = {\n")
    w = DCE_net.state_dict()[weight_list[4]].to(device).numpy()
    w = w.flatten()
    count = 0
    for param in w:
        count += 1
        f.write(str(int(param * 16384)))

        if count != w.size:
            f.write(", ")
        else:
            f.write("};\n")

        if count % 16 == 0:
            f.write("\n")
    f.write("\n")

    print("export ", weight_list[5])
    f.write("const short conv3_b[3] = {\n")
    w = DCE_net.state_dict()[weight_list[5]].to(device).numpy()
    w = w.flatten()
    count = 0
    for param in w:
        count += 1
        f.write(str(int(param * 16384 * 16384)))

        if count != w.size:
            f.write(", ")
        else:
            f.write("};\n")

        if count % 8 == 0:
            f.write("\n")
    f.write("\n")
