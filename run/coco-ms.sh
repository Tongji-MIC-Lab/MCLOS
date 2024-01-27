#!/bin/bash

port=$(python get_free_port.py)
echo ${port}
alias exp="CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port ${port} --nproc_per_node=2 run_ddp.py --num_workers 4"
# shopt -s expand_aliases

ds=coco
task=$1  # 20-0, 20-1, 20-2, 20-3

# if you performed them for the SS experiment, comment the following!
exp --method FT --name FT --epochs 20 --lr 0.01 --batch_size 12
exp --method COS --name COS --epochs 20 --lr 0.01 --batch_size 12
exp --method SPN --name SPN --epochs 20 --lr 0.01 --batch_size 12
exp --method DWI --name DWI --epochs 20 --lr 0.01 --batch_size 12 --ckpt checkpoints/step/${task}-voc/COS_0.pth
exp --method RT --name RT --epochs 40 --lr 0.01 --batch_size 12 --ckpt checkpoints/step/${task}-voc/FT_0.pth --born_again

gen_par="--task ${task} --dataset ${ds} --batch_size 5"
lr=0.001
iter=1000
path=checkpoints/step/${task}-${ds}


# for 1 shot, disable the batch norm pooling on the deeplab head (or it'll throw errors) (--no_pooling)
for is in 0 1 2; do
    inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5 --no_pooling"
    ns=1

    exp --method FT --name FT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth

    exp --method WI --name WI --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth
    exp --method DWI --name DWI --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/DWI_0.pth
    exp --method RT --name RT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/RT_0.pth

    exp --method SPN --name SPN --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/SPN_0.pth
    exp --method AMP --name AMP --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth

    exp --method LWF --name LWF --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
    exp --method ILT --name ILT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
    exp --method MIB --name MIB --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
    exp --method PIFS --name PIFS --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth

    exp --method MCLOS --name MCLOS --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth --warm_epoch 10 --contrast_weight 0.1 --pixel_update_freq 50

    for s in 2 3 4 5; do
      exp --method FT --name FT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}

      exp --method WI --name WI --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
      exp --method DWI --name DWI --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
      exp --method RT --name RT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}

      exp --method SPN --name SPN --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
      exp --method AMP --name AMP --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}

      exp --method LWF --name LWF --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
      exp --method ILT --name ILT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
      exp --method MIB --name MIB --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
      exp --method PIFS --name PIFS --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
      
      exp --method MCLOS --name MCLOS --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth --warm_epoch 10 --contrast_weight 0.1 --pixel_update_freq 50
    done
done

for ns in 2 5; do
  for is in 0 1 2; do
      inc_par="--ishot ${is} --input_mix novel --val_interval 1000 --ckpt_interval 5"

      exp --method FT --name FT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth

      exp --method WI --name WI --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth
      exp --method DWI --name DWI --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/DWI_0.pth
      exp --method RT --name RT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/RT_0.pth

      exp --method SPN --name SPN --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/SPN_0.pth
      exp --method AMP --name AMP --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth

      exp --method LWF --name LWF --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
      exp --method ILT --name ILT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
      exp --method MIB --name MIB --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth
      exp --method PIFS --name PIFS --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth

      exp --method MCLOS --name MCLOS --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth --warm_epoch 10 --contrast_weight 0.1 --pixel_update_freq 50

      for s in 2 3 4 5; do
        exp --method FT --name FT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}

        exp --method WI --name WI --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
        exp --method DWI --name DWI --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
        exp --method RT --name RT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}

        exp --method SPN --name SPN --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
        exp --method AMP --name AMP --iter 0 --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}

        exp --method LWF --name LWF --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
        exp --method ILT --name ILT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
        exp --method MIB --name MIB --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}
        exp --method PIFS --name PIFS --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step $s --nshot ${ns}

        exp --method MCLOS --name MCLOS --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/COS_0.pth --warm_epoch 10 --contrast_weight 0.1 --pixel_update_freq 50
      done
  done
done
