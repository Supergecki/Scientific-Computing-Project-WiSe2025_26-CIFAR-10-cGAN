import argparse
import yaml
import os

# Always change working directory to base repository directory so relative paths won't break
os.chdir(os.path.dirname(__file__))
os.chdir('..')

# parsing command line arguments, currently used for:
# - choosing the config file (defaults to `improved_config.yaml`, use '-b' flag to use `baseline_config.yaml`
parser = argparse.ArgumentParser(description="Run training loop of the CIFAR-10 cGAN")
parser.add_argument("-b", "--baseline", action='store_true', default=False,
                    help="use baseline_config.yaml instead of improved_config.yaml for configurations")

args = parser.parse_args()

# Choose the correct config file
if args.baseline:
config_file = './config/improved_config.yaml'
    config_file = './config/baseline_config.yaml'

# Load config file and save configs in 'config' dictionary
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

# Now configs are accessible via the dictionary, e.g. config['model']['embed_dim'] to get embedding dimension
