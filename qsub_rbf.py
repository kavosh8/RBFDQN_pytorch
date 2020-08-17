import os,sys,re,time
import os.path

bash_script = '''#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -t 48:00:00
source ~/anaconda3/bin/activate pyt3
echo "prog started at: $(date)"
cd ~/RBFDQN_pytorch
module load opengl/mesa-12.0.6
module load ffmpeg/4.0.1
python RBFDQN.py {} {}
'''
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH/home/kasadiat/.mujoco/mujoco200/bin s
if not os.path.exists('pbs_files'):
	os.makedirs('pbs_files')






min_seed=0
max_seed=5

for seed_num in range(min_seed,max_seed):
	for domain in [4]:
		for setting in range(2,10):
			hyper_parameter_name=domain*10+setting
			outfile="pbs_files/RBFDQN{}_{}.pbs".format(str(hyper_parameter_name),
														  str(seed_num)
														 )
			output=open(outfile, 'w')
			output.write(bash_script.format(str(hyper_parameter_name),str(seed_num)))
												
			output.close()
			cmd="sbatch {}".format(outfile)
			os.system(cmd)
			time.sleep(.01)
