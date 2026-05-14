import subprocess

python_bin = (
    "/gpfs3/well/papiez/users/zwk579/.conda_envs/disen_py39/bin/python"
)

command = [
    python_bin,
    "-m",
    "src.train",
    "-c",
    "configs/config_ukb.yaml",
    "--resume",
    "/well/papiez/users/zwk579/Analysis/disentangling-retinal-images/outputs/2026-05-10/ukb_vessel_dcor/checkpoints/_epoch=53.ckpt",
]

subprocess.run(command)