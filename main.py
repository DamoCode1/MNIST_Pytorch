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
    "none": nn.Identity
}

class classifier(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28 * 28, 28 * 28)
        self.activation1 = ACTIVATIONS[activation]()
        self.linear2 = nn.Linear(28 * 28, 10)

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

def toTensorDataset(split):
    x = torch.stack([transform(img) for img in split["image"]])
    y = torch.tensor(split["label"])
    return torch.utils.data.TensorDataset(x, y)

rawDataset = load_dataset("ylecun/mnist")
trainDataset = toTensorDataset(rawDataset["train"])
valDataset = toTensorDataset(rawDataset["test"].train_test_split(test_size = 0.5, seed = 67)["train"])  # same val split as before

def train(model, epochCount, activationName, curSeed):
    curGen = torch.Generator().manual_seed(curSeed)
    trainingLoader = torch.utils.data.DataLoader(trainDataset, batch_size = 256, shuffle = True, generator = curGen)
    validatingLoader = torch.utils.data.DataLoader(valDataset, batch_size = 256)

    lossFunction = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr = 1e-4)
    earlyStopping = EarlyStopping(patience=5, verbose=True, path = "earlyStop.pth")

    print(f"Beginning training of {activationName} with seed = {curSeed}")
    writer = SummaryWriter(f"runs/{activationName}_{curSeed}")
    for i in range(0, epochCount):
        model.train()
        trainTotal = 0.0
        trainCorrect = 0.0
        trainLoss = 0.0
        for inputs, labels in trainingLoader:
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
            trainLoss += loss.item() * len(labels)
        model.eval()
        validateTotal = 0.0
        validateCorrect = 0.0
        validateLoss = 0.0
        for inputs, labels in validatingLoader:
            with torch.no_grad():
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
        writer.add_scalar("Loss/train", trainLoss / trainTotal, i)
        writer.add_scalar("Loss/val", validateLoss / validateTotal, i)

        earlyStopping(validateLoss, model)
        if earlyStopping.early_stop:
            print(f"{activationName} (Seed = {curSeed}): early stopping triggered at epoch {i}")
            break
    writer.close()
    model.load_state_dict(torch.load("earlyStop.pth"))
    torch.save(model.state_dict(), f"model_{activationName}_{curSeed}.pth")

def test(model, testCount, runName, curSeed):
    hfDataset = load_dataset("ylecun/mnist")
    testDataset = hfDataset["test"].shuffle(seed = curSeed).select(range(testCount)).train_test_split(test_size = 0.5, seed = curSeed)["test"]  # Splits test set into validation (50%) and test (50%) sets
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

        fig.suptitle(f"{runName} (seed = {curSeed}): Sample {j + 1}/{testCount}")
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.xpu.is_available())
    print(torch.xpu.get_device_name(0))
    if input("Retrain model? Y/N") == 'Y':
        if input("Run activations experiment? Y/N") == 'Y':
            for activation in list(ACTIVATIONS):
                for i in range(0, 5):
                    torch.manual_seed(i)
                    model = classifier(activation).to(device)
                    train(model, epochCount = 50, activationName = activation, curSeed = i)
                    test(model, testCount = 5, runName = activation, curSeed = i)
        else:
            activation = input("Which activation function? (See documentation)")
            chosenSeed = input("Which seed? (0-4)")
            model = classifier(activation).to(device)
            train(model, epochCount = 50, activationName = activation, curSeed = chosenSeed)
            test(model, testCount = 20, runName = activation, curSeed = chosenSeed)
    else:
        activation = input("Which activation function? (See documentation)")
        chosenSeed = input("Which seed? (0-4)")
        model = classifier(activation).to(device)
        model.load_state_dict(torch.load(f"model_{activation}.pth", map_location = device))
        test(model, testCount = 20, runName = activation, curSeed = chosenSeed)
