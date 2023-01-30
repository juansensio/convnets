from .alexnet import alexnet 

def vgg():
	config = alexnet()
	config.update(
		model='VGG',
		dropout=0.9,
		batch_size=256,
		epochs=74
	)
	return config
	