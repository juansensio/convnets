import argparse
import convnets.train.imagenet.configs as configs
import convnets.models as models

def train(config):
	print(config)
	model = getattr(models, config['model'])(config)
	print(model)
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process imagenet.')
	parser.add_argument('--base-config', help='Base configuration to be used for training', default="alexnet")
	args = parser.parse_args()
	config = getattr(configs, args.base_config)()
	train(config)