import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from datasets import DatasetDict, load_dataset

#TODO:
# Create basic model
# Add Intel GPU support (Complete)
# Try adding normalisation
# Automatic early stopper, something like: (early_stopper = EarlyStopping(patience=5, min_delta=0.001, relative_drop_limit=0.10))
# Create UI that allows model to be trained with certain function (or experiment with multiple below), then if just 1 allow visualise
# Create system that runs model but for multiple different activation functions
# Make practice paper on different types of activation functions impact on results

class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28 * 28, 100 * 100)
        self.ReLU1 = nn.ReLU()
        self.linear2 = nn.Linear(100 * 100, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.ReLU1(x)
        x = self.linear2(x)
        return x

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
global model

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

def applyTransform(batch):
    batch["image"] = [transform(img) for img in batch["image"]]
    return batch

def train(model, epochCount = 10):
    hfDataset = load_dataset("ylecun/mnist")
    valDataset = hfDataset["test"].train_test_split(test_size=0.5, seed=67)["train"] #Splits test set into validation (50%) and test (50%) sets
    hfDataset["train"].set_transform(applyTransform)
    valDataset.set_transform(applyTransform)

    trainingLoader = torch.utils.data.DataLoader(hfDataset["train"], batch_size = 256, shuffle = True, num_workers = 4, pin_memory = True)
    validatingLoader = torch.utils.data.DataLoader(valDataset, batch_size = 256, shuffle = True, num_workers = 4, pin_memory = True)

    lossFunction = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr = 1e-4)

    print("Beggining training")
    writer = SummaryWriter(f"runs/base")
    for i in range(0, epochCount):
        model.train()
        trainTotal = 0.0
        trainCorrect = 0.0
        for j, data in enumerate(trainingLoader):
            inputs, labels = data["image"], data["label"]
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimiser.zero_grad() #Set all gradients to 0
            output = model(inputs)
            loss = lossFunction(output, labels)
            loss.backward()
            optimiser.step()
            prediction = output.argmax(1)
            trainTotal += len(labels)
            trainCorrect += (prediction == labels).sum().item()
        model.eval()
        validateTotal = 0.0
        validateCorrect = 0.0
        for j, data in enumerate(validatingLoader):
            inputs, labels = data["image"], data["label"]
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            prediction = output.argmax(1)
            validateTotal += len(labels)
            validateCorrect += (prediction == labels).sum().item()

        trainAccuracy = (trainCorrect / trainTotal) * 100
        validateAccuracy = (validateCorrect / validateTotal) * 100
        writer.add_scalar("Accuracy/train", trainAccuracy, i)
        writer.add_scalar("Accuracy/val", validateAccuracy, i)
        print(f"Epoch {i} had testing accuracy of {trainAccuracy}% and validating accuracy of {validateAccuracy}%")
    writer.close()



if __name__ == '__main__':
    print(torch.__version__)
    print(torch.xpu.is_available())
    print(torch.xpu.get_device_name(0))
    model = classifier().to(device)
    train(model)
