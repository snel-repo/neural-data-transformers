# Src: Andrew's tune_tf2

from os import path

repo_path = path.dirname(path.realpath(__file__))
DEFAULT_CONFIG_DIR = path.join(repo_path, 'configs')

# contains data about general status of PBT optimization
PBT_CSV = 'pbt_state.csv'
# contains data about which models are exploited
EXPLOIT_CSV = 'exploits.csv'
# contains data about which hyperparameters are used
HPS_CSV= 'gen_config.csv'
