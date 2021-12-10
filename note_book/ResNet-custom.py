# load up the ResNet50 model
model = resnet50(pretrained=True)
numFeatures = model.fc.in_features
# loop over the modules of the model and set the parameters of
# batch normalization modules as not trainable
for module, param in zip(model.modules(), model.parameters()):
	if isinstance(module, nn.BatchNorm2d):
		param.requires_grad = False
# define the network head and attach it to the model
headModel = nn.Sequential(
	nn.Linear(numFeatures, 512),
	nn.ReLU(),
	nn.Dropout(0.25),
	nn.Linear(512, 256),
	nn.ReLU(),
	nn.Dropout(0.5),
	nn.Linear(256, len(trainDS.classes))
)
model.fc = headModel
# append a new classification top to our feature extractor and pop it
# on to the current device
model = model.to(config.DEVICE)