'''
python eval_all.py --model_path ./weights/{}.pth --detect_method {CNNSpot,Gram,Fusing,FreDect,LGrad,LNP,DIRE}  --noise_type {blur,jpg,resize}
'''

import os
import csv
import torch

from validate import validate,validate_single
from options import TestOptions
from eval_config import *
from PIL import ImageFile
from util import create_argparser,get_model, set_random_seed



ImageFile.LOAD_TRUNCATED_IMAGES = True








# 固定随机种子
set_random_seed()
# Running tests


opt = TestOptions().parse(print_options=True) #获取参数类
opt.dataset_name = 'ELSA'


model_name = os.path.basename(opt.model_path).replace('.pth', '')
results_dir=f"./results/{opt.detect_method}"
mkdir(results_dir)

rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]

print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)

    # model = resnet50(num_classes=1)
    model = get_model(opt)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    try:
        if opt.detect_method in ["FreDect","Gram"]:
            model.load_state_dict(state_dict['netC'],strict=True)
        elif opt.detect_method == "UnivFD":
            model.fc.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict['model'],strict=True)
    except:
        print("[ERROR] model.load_state_dict() error")
    model.cuda()
    model.eval()


    opt.process_device=torch.device("cpu")
    acc, ap, _, _, _, _ = validate(model, opt)
    rows.append([val, acc, ap])
    print("({}) acc: {}; ap: {}".format(val, acc, ap))


# 结果文件
csv_name = results_dir + '/{}_{}.csv'.format(opt.detect_method,opt.noise_type)
with open(csv_name, 'a+') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
