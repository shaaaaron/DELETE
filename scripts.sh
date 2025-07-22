# Train and Retrain From Scratch
python main.py --train_from_scratch --retrain_from_scratch

# Unlearn on CIFAR10
# Most of the methods here are configured with their recommended (or best) settings. If the performance is not as expected, you may need to adjust the parameters as described in the Supplementary Material.
CUDA_VISIBLE_DEVICES=0 python main.py  --method random_label    --unlearn_rate 3e-4  --description lr_3e-4
CUDA_VISIBLE_DEVICES=0 python main.py  --method finetune        --unlearn_rate 2e-2  --description lr_2e-2
CUDA_VISIBLE_DEVICES=0 python main.py  --method gradient_ascent --unlearn_rate 1e-5  --description lr_1e-5
CUDA_VISIBLE_DEVICES=0 python main.py  --method boundary_shrink --unlearn_rate 5e-5  --description lr_5e-5
CUDA_VISIBLE_DEVICES=0 python main.py  --method boundary_expand --unlearn_rate 8e-4  --description lr_8e-4
CUDA_VISIBLE_DEVICES=2 python main.py  --method l2ul_adv        --unlearn_rate 1e-5  --description lr_1e-5
CUDA_VISIBLE_DEVICES=2 python main.py  --method fisher          --alpha        1e-7  --description lr_1e-7
CUDA_VISIBLE_DEVICES=2 python main.py  --method wood_fisher     --alpha        1     --description lr_1
CUDA_VISIBLE_DEVICES=1 python main.py  --method delete          --unlearn_rate 1e-3  --description lr_1e-3
