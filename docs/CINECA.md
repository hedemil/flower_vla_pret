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

# Debug
sbatch $LEONARDO_FAST/project/flower_vla_pret/scripts/leonardo/sbatch_debug.sh
# Train
sbatch $LEONARDO_FAST/project/flower_vla_pret/scripts/leonardo/sbatch_debug.sh
```

## Watch the logs
```bash
squeue -u $USER
# Once it starts tail the output
# Debug
cat flowervla-dbg_*.out
# Train
cat flowervla_*.out
cat flowervla_*.err
```

## Sync with wandb
```bash
source $LEONARDO_WORK/project/venvs/flowervla/bin/activate
wandb login
export LEONARDO_FAST=/leonardo_scratch/fast/AIFAC_P01_047
bash $LEONARDO_FAST/project/flower_vla_pret/scripts/leonardo/sync_wandb.sh
```


