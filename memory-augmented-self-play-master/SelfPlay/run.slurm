#!/bin/bash
#SBATCH -J test            # Job 
#SBATCH -o ./test.out  # 输出
#SBATCH --qos=normal       # qos(quality of service): normal or debug, 对应不同优先级及最大可用时长.
#SBATCH -p geforce         # 指定 partition: geforce,k80,etc.
#SBATCH --nodelist=gpu02   # 指定属于上述 partition 的特定节点.也可删除这一行,由系统自动分配.
#SBATCH --cpus-per-task=2  # 申请 cpu processor 数; 可用内存与申请 cpu processor 数成正比.
#SBATCH --gres=gpu:1       # 申请 gpu 数
#SBATCH -N 1               # 申请节点数,一般为1
#SBATCH -t 10:00:00        # 申请 Job 运行时长0小时5分钟0秒,若要申请一天时间以上,如申请1天,书写格式为#SBATCH -t 1-00:00:00

# 上述每一项 SBATCH 参数不指定时系统会指定一个默认值

# 随着 Job 的提交和执行，slurm 会帮助用户在申请的节点上挨个执行下述命令
module add cuda/9.0
module add anaconda/3
python app/mountaincar_reinforce.py --model_num_epochs 1000 --model_batch_size 10 --model_max_steps_per_episode 100 --model_is_self_play False> log.txt
#python app/acrobot_reinforce.py --model_num_epochs 6 --model_batch_size 2 --model_max_steps_per_episode 100 --model_is_self_play False > log.txt
#python app/mazebase_reinforce.py --model_num_epochs 6 --model_batch_size 2 --model_use_baseline False --model_max_steps_per_episode 100 --model_is_self_play False > log.txt
#python app/cartpole_reinforce.py --model_num_epochs 6 --model_batch_size 2 --model_max_steps_per_episode 100 --model_use_baseline True --model_is_self_play False
#python app/selfplay.py --model_num_epochs 3 --model_batch_size 2 --model_max_steps_per_episode 40  --model_is_self_play True --model_is_self_play_with_memory True --model_memory_type lstm_memory > log.txt
#python app/selfplay.py --model_num_epochs 3 --model_batch_size 2 --model_max_steps_per_episode 40  --model_is_self_play True --model_is_self_play_with_memory True --model_memory_type base_memory > log.txt
#python app/selfplay.py --model_num_epochs 3 --model_batch_size 2 --model_max_steps_per_episode 40 --model_is_self_play True > log.txt
python app/plot.py
