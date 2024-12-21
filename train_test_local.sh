name="X_FPN_ASPP_conv_12_24_32_HEAD_Z0_xy"
UAVhw=96
Satellitehw=384
batchsize=16
num_epochs=40
centerR=33
lr=0.0003
backbone="pcpvt_small"
neg_weight=15.0
share=0
python train.py --name $name --centerR $centerR --UAVhw $UAVhw --Satellitehw $Satellitehw \
                --backbone $backbone --batchsize $batchsize \
                --batchsize $batchsize --num_epochs $num_epochs --lr $lr --neg_weight $neg_weight --share $share