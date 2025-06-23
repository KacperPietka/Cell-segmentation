import segmentation_models as sm
import torch
import segmentation_models_pytorch as smp
from datasets import load_dataset

dataset = load_dataset("jxmorris12/livecell")



x_train, y_train, x_val, y_val = dataset

model = sm.Unet('resnet34', encoder_weights='imagenet', classes=1, activation='sigmoid')

model.fit(x=x_train, y=y_train, epochs=100, batch_size=16, validation_data=(x_val,y_val))


