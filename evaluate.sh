#! /bin/bash

for i in {1..15}
do
    CUDA_VISIBLE_DEVICES=0 vot evaluate --workspace /path/to/your/workspace  LY_swinb_dm_deaot_vots
done

vot pack --workspace /path/to/your/workspace  LY_swinb_dm_deaot_vots

