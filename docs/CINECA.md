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
tmux new -s train
source $LEONARDO_WORK/project/venvs/flowervla/bin/activate
cd $LEONARDO_FAST/project/flower_vla_pret
# Debug
sbatch $LEONARDO_FAST/project/flower_vla_pret/scripts/leonardo/sbatch_debug.sh
# Train
sbatch $LEONARDO_FAST/project/flower_vla_pret/scripts/leonardo/sbatch_train.sh
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


# Download checkpoint
```bash
# Checkpoint
rsync -avP leonardo:/leonardo_scratch/fast/AIFAC_P01_047/project/output/checkpoints/runs/2026-02-27/09-40-45/checkpoint_20000 ./

# Hydra
rsync -avP leonardo:/leonardo_scratch/fast/AIFAC_P01_047/project/output/checkpoints/runs/2026-02-27/09-40-45/.hydra ./
```
