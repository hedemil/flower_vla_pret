# Re-generate certificate after 12 hrs
```bash
step ssh login 'emhed@kth.se' --provisioner cineca-hpc
```

## Login to LEONARDO
```bash
ssh ehed0000@login.leonardo.cineca.it
```
## Start tmux session for training
```bash
tmux new -s <sessionname>
source $LEONARDO_WORK/project/venvs/flowervla/bin/activate
export LEONARDO_DATA_DIR=$LEONARDO_WORK/project/data/tensorflow_datasets
export LEONARDO_FAST=/leonardo_scratch/fast/AIFAC_P01_047
export LEONARDO_WORK=/leonardo_work/AIFAC_P01_047

sbatch $LEONARDO_FAST/project/flower_vla_pret/scripts/leonardo/sbatch_debug.sh
```

## Watch the logs
```bash
squeue -u $USER
# Once it starts tail the output
cat flowervla-dbg_*.out
```

## Sync with wandb
```bash
source $LEONARDO_WORK/project/venvs/flowervla/bin/activate
wandb login
export LEONARDO_FAST=/leonardo_scratch/fast/AIFAC_P01_047
bash $LEONARDO_FAST/project/flower_vla_pret/scripts/leonardo/sync_wandb.sh
```


