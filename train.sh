# ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 nohup torchrun --nproc_per_node=4 train_sft.py > ttt.out &
# nohup torchrun --nproc_per_node=16 train_onlinedpo1.py > qwen2.5vl3b.out &
# nohup torchrun --nproc_per_node=8 train_grpo.py --loss_type grpo --output_dir output_grpo_qwen2_5vl_7b > qwen2.5vl7b-grpo.out &
# torchrun --nproc_per_node=8 train_grpo.py --loss_type grpo --output_dir output_grpo_qwen2_5vl_7b > qwen2.5vl7b-grpo.out
# torchrun --nproc_per_node=8 train_grpo.py --loss_type dapo --output_dir output_dapo_qwen2_5vl_7b > qwen2.5vl7b-dapo.out
# torchrun --nproc_per_node=8 train_grpo.py --loss_type dr_grpo --output_dir output_dr_qwen2_5vl_7b > qwen2.5vl7b-dr_grpo.out
# torchrun --nproc_per_node=8 train_sft.py > sft.out
# nohup torchrun --nproc_per_node=8 train_grpo.py --loss_type grpo --output_dir output_grpo_qwen2_5vl_3b > qwen2.5vl3b-grpo.out &
# torchrun --nproc_per_node=8 train_grpo.py --loss_type dapo --model_path ../Downloads/Models/Qwen/Qwen2.5-VL-7B-Instruct --output_dir output_dapo_qwen2_5vl_7b > qwen2.5vl7b-dapo.out
# torchrun --nproc_per_node=8 train_grpo.py --loss_type dr_grpo --model_path ../Downloads/Models/Qwen/Qwen2.5-VL-7B-Instruct --output_dir output_dr_qwen2_5vl_7b > qwen2.5vl7b-dr_grpo.out
# torchrun --nproc_per_node=8 train_grpo.py --loss_type grpo --model_path ../Downloads/Models/Qwen/Qwen2.5-VL-7B-Instruct --output_dir output_grpo_qwen2_5vl_7b > qwen2.5vl7b-grpo.out 
# nohup torchrun --nproc_per_node=8 train_sft.py > sftr16.out &
# nohup torchrun --nproc_per_node=8 train_sft.py --model_path ../Downloads/Models/Qwen/Qwen2.5-VL-3B-Instruct --mode sft --output_dir output_sft_qwen2_5vl_3bR16 > qwen2.5vl-sft3bR16.out &
# torchrun --nproc_per_node=8 train_sft.py --model_path ../Downloads/Models/Qwen/Qwen2.5-VL-3B-Instruct --mode syndata --output_dir output_syndata_qwen2_5vl_3bR16 > qwen2.5vl-syndata3bR16.out
# torchrun --nproc_per_node=8 train_sft.py --model_path ../Downloads/Models/Qwen/Qwen2.5-VL-7B-Instruct --mode sft --output_dir output_sft_qwen2_5vl_7bR16 > qwen2.5vl-sft7bR16.out
# torchrun --nproc_per_node=8 train_sft.py --model_path ../Downloads/Models/Qwen/Qwen2.5-VL-7B-Instruct --mode syndata --output_dir output_syndata_qwen2_5vl_7bR16 > qwen2.5vl-syndata7bR16.out
# nohup torchrun --nproc_per_node=8 train_papo.py --loss_type dapo --output_dir output_papod_qwen2_5vl_3b > qwen2.5vl3b-papod.out &
# torchrun --nproc_per_node=8 train_papo.py --model_path ../Downloads/Models/Qwen/Qwen2.5-VL-7B-Instruct --loss_type grpo --output_dir output_papog_qwen2_5vl_7b > qwen2.5vl7b-papog.out
# nohup torchrun --nproc_per_node=8 train_papo.py --model_path ../Downloads/Models/Qwen/Qwen2.5-VL-3B-Instruct --loss_type dapo --output_dir output_papod_qwen2_5vl_3b > qwen2.5vl3b-papod.out &
# nohup torchrun --nproc_per_node=8 train_papo.py --model_path ../Downloads/Models/Qwen/Qwen2.5-VL-3B-Instruct --loss_type grpo --output_dir output_papog_off_qwen2_5vl_3b > qwen2.5vl3b-papog-off.out &

torchrun --nproc_per_node=8 train_rtpo.py --model_path ../Downloads/Models/Qwen/Qwen2.5-VL-3B-Instruct --loss_type grpo    --output_dir output_rtpoGsyn2_5vl_3b  > qwen2.5vl3b-rtpoGsyn.out  --think_process_key explanation
torchrun --nproc_per_node=8 train_rtpo.py --model_path ../Downloads/Models/Qwen/Qwen2.5-VL-3B-Instruct --loss_type dr_grpo --output_dir output_rtpoDRsyn2_5vl_3b > qwen2.5vl3b-rtpoDRsyn.out --think_process_key explanation
torchrun --nproc_per_node=8 train_rtpo.py --model_path ../Downloads/Models/Qwen/Qwen2.5-VL-3B-Instruct --loss_type dapo    --output_dir output_rtpoDsyn2_5vl_3b  > qwen2.5vl3b-rtpoDsyn.out  --think_process_key explanation