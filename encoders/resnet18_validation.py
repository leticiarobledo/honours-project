import time
import torch
import torch.nn as nn
import torch.nn.functional as F
#if torch.cuda.is_available():
#    torch.backends.cudnn.deterministic = True

##########################
### SETTINGS
##########################

print("setting hyperarameters")

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10
# Architecture
NUM_FEATURES = 28*28
NUM_CLASSES = 10
# Other
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAYSCALE = True
print(DEVICE)

##########################
###     ALL DATASETS
##########################

print("TRAINING RESNET18 ON --ALL-- DATASETS. USPS is FLOAT")
print("DEFAULT RESNET ARCHITECTURE with --VALIDATION--. NO PARALLELISM. ")

from data_preprocessing import get_datasets_validation_no_svhn #get_datasets_validation
train_loader, val_loader, test_loader = get_datasets_validation_no_svhn("dataset", BATCH_SIZE)


##########################
### MODEL
##########################
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        #x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

def resnet18(num_classes):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[2, 2, 2, 2],
                   num_classes=NUM_CLASSES,
                   grayscale=GRAYSCALE)
    return model


#####################################
#####        INITIALISE     #########
torch.manual_seed(RANDOM_SEED)

model = resnet18(NUM_CLASSES)
# model = model.double()
# model= nn.DataParallel(model)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  


######################################
######      TRAINING        ###########
# Check accuracy on training & test to see how good our model
def check_accuracy(model, loader):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()

    return correct_pred.float()/num_examples

loss = []
path = "resnet_default_val_no_svhn.txt"
best_val_acc = 0
best_model = None
with open(path, "w") as f:
    for epoch in range(NUM_EPOCHS):
        #continue
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
                
            ### FORWARD AND BACK PROP
            logits, probas = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            
            cost.backward()
            loss.append(cost)

            ### UPDATE MODEL PARAMETERS
            optimizer.step()
          
        model.eval()
        with torch.set_grad_enabled(False):
            val_acc = check_accuracy(model, val_loader)
            train_acc = check_accuracy(model, train_loader)
            test_acc = check_accuracy(model, test_loader)
            # If the current model has a better validation accuracy, save it
            if val_acc > best_val_acc:
                best_model = model.state_dict()
                best_val_acc = val_acc

            # Check accuracy on training & test to see how good our model
            print("EPOCH: " + str(epoch))
            print(f"Accuracy on training set: {train_acc*100:.2f}")
            print(f"Accuracy on validation set: {val_acc*100:.2f}")
            print(f"Accuracy on test set: {test_acc*100:.2f}")
            f.write("EPOCH " + str(epoch) + ", train_acc:" + str(train_acc) + 
                    ", val_acc:" + str(val_acc) + ", test_acc:" + str(test_acc) + "\n")


def save_checkpoint(state, filename):
    name = filename + ".pth.tar"
    print("=> Saving checkpoint")
    torch.save(state, name)

# save model
checkpoint = {"state_dict": best_model, "optimizer": optimizer.state_dict()}
# Try save checkpoint
save_checkpoint(checkpoint, filename="resnet18_default_val_no_svhn")
