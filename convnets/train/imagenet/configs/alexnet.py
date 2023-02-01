def alexnet():
	return {
		'model': 'Alexnet',
		'optimizer': 'SGD',
		'optimizer_params': {
			'lr': 1e-2,
			'momentum': 0.9,
			'weight_decay': 0.0005,
		},
		'scheduler': 'ReduceLROnPlateau',
		'scheduler_params': {
			'patience': 1,
			'gamma': 0.1,
		},
		'batch_size': 128,
		'epochs': 90,
		'gpus': 1,
		'transforms': {
			'RandomCrop': {
				'width': 224,
				'height': 224
			},
			'HorizontalFlip': {},
			'RGBShift': {}
		},
		'path': '/fastdata/imagenet256',
	}