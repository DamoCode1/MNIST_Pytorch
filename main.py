import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision.transforms import v2
from datasets import DatasetDict, load_dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from early_stopping_pytorch import EarlyStopping

#TODO:
# Create basic model (Complete)
# Add Intel GPU support (Complete)
# Try adding normalisation (Input normalisation, no batch normalisation) (1/2 Complete)
# Automatic early stopper, something like: (early_stopper = EarlyStopping(patience=5, min_delta=0.001, relative_drop_limit=0.10)) (Complete)
# Create UI that allows model to be trained with certain function (or experiment with multiple below), then if just 1 allow visualise (Complete, terminal for controls, matplotlib for test images, tensor board for experiment)
# Create system that runs model but for multiple different activation functions (Complete)
# Make practice paper on different types of activation functions impact on results

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
}

class classifier(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28 * 28, 100 * 100)
        self.activation1 = ACTIVATIONS[activation]()
        self.linear2 = nn.Linear(100 * 100, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        return x

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    v2.Normalize((0.1307,), (0.3081,)) #https://github.com/pytorch/examples/blob/main/mnist/main.py
])

def applyTransform(batch):
    batch["image"] = [transform(img) for img in batch["image"]]
    return batch

def train(model, epochCount, activationName):
    hfDataset = load_dataset("ylecun/mnist")
    valDataset = hfDataset["test"].train_test_split(test_size = 0.5, seed = 67)["train"] #Splits test set into validation (50%) and test (50%) sets
    hfDataset["train"].set_transform(applyTransform)
    valDataset.set_transform(applyTransform)

    trainingLoader = torch.utils.data.DataLoader(hfDataset["train"], batch_size = 256, shuffle = True, num_workers = 4, pin_memory = True)
    validatingLoader = torch.utils.data.DataLoader(valDataset, batch_size = 256, shuffle = True, num_workers = 4, pin_memory = True)

    lossFunction = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr = 1e-4)
    earlyStopping = EarlyStopping(patience=5, verbose=True)

    print("Beginning training")
    writer = SummaryWriter(f"runs/{activationName}")
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
        validateLoss = 0.0
        for j, data in enumerate(validatingLoader):
            inputs, labels = data["image"], data["label"]
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            prediction = output.argmax(1)
            validateLoss += lossFunction(output, labels).item() * len(labels)
            validateTotal += len(labels)
            validateCorrect += (prediction == labels).sum().item()

        trainAccuracy = (trainCorrect / trainTotal) * 100
        validateAccuracy = (validateCorrect / validateTotal) * 100
        writer.add_scalar("Accuracy/train", trainAccuracy, i)
        writer.add_scalar("Accuracy/val", validateAccuracy, i)
        print(f"{activationName}'s epoch {i} had testing accuracy of {trainAccuracy}% and validating accuracy of {validateAccuracy}%")

        earlyStopping(validateLoss, model)
        if earlyStopping.early_stop:
            print(f"{activationName}: early stopping triggered at epoch {i}")
            break
    writer.close()
    torch.save(model.state_dict(), "model.pth")

def test(model, testCount, runName):
    hfDataset = load_dataset("ylecun/mnist")
    testDataset = hfDataset["test"].shuffle(seed = 67).select(range(testCount)).train_test_split(test_size = 0.5, seed = 67)["test"]  # Splits test set into validation (50%) and test (50%) sets
    testDataset.set_transform(applyTransform)
    testingLoader = torch.utils.data.DataLoader(testDataset, batch_size = 1)

    model.eval()
    for j, data in enumerate(testingLoader):
        inputs, labels = data["image"], data["label"]
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)
        probabilities = F.softmax(output, dim=1)[0].detach().cpu()
        label = labels.item()
        plt.ion()
        fig, (axImg, axBar) = plt.subplots(1, 2, figsize=(9, 4))
        axImg.clear()
        axImg.imshow(inputs[0].squeeze().cpu(), cmap="gray")
        axImg.axis("off")

        axBar.clear()
        axBar.bar(range(10), probabilities * 100)
        axBar.set_xticks(range(10))
        axBar.set_ylim(0, 100)
        axBar.set_xlabel("Digit")
        axBar.set_ylabel("Confidence (%)")

        fig.suptitle(f"{runName} - Sample {j + 1}/{testCount}")
        plt.ioff()
        plt.show()



if __name__ == '__main__':
    print(torch.__version__)
    print(torch.xpu.is_available())
    print(torch.xpu.get_device_name(0))
    if input("Retrain model? Y/N") == 'Y':
        if input("Run activations experiment? Y/N") == 'Y':
            for activation in list(ACTIVATIONS):
                model = classifier(activation).to(device)
                train(model, epochCount = 50, activationName = activation)
                test(model, testCount = 5, runName = activation)
        else:
            model = classifier("relu").to(device)
            train(model, epochCount = 50, activationName = "relu")
    else:
        model = classifier("relu").to(device)
        model.load_state_dict(torch.load("model.pth", map_location = device))
        model.eval()
    test(model)