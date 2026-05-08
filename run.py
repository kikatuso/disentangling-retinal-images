import os

python_bin = (
    "/gpfs3/well/papiez/users/zwk579/.conda_envs/disen_py39/bin/python"
)

command = f"{python_bin} -m src.train -c configs/config_ukb.yaml"

os.system(command)