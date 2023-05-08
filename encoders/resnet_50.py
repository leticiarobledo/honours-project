import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

#if torch.cuda.is_available():
#    torch.backends.cudnn.deterministic = True

##########################
### SETTINGS
##########################

print("setting hyperarameters")
# Hyperparameters
RANDOM_SEED = 49
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 30
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

print("TRAINING RESNET50 ON --ALL-- DATASETS. USPS is FLOAT")
print("NO-PRETRAINED RESNET ARCHITECTURE. NO PARALLELISM. ")

from data_preprocessing import get_datasets_validation
train_loader, val_loader, test_loader = get_datasets_validation("dataset", BATCH_SIZE)

#####################################
#####        INITIALISE     #########

torch.manual_seed(RANDOM_SEED)

model = models.resnet50(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model.avgpool = nn.Identity()

# model = model.double()
# model= nn.DataParallel(model)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  
criterion = nn.CrossEntropyLoss()


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^

######################################
######      TRAINING        ###########
# Check accuracy on training & test to see how good our model
def check_accuracy(model, loader):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(features)
        _, predicted_labels = torch.max(outputs, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()

    return correct_pred.float()/num_examples

loss = []
path = "resnet_50_val_pretrained.txt"
best_val_acc = 0
best_model = None
with open(path, "w") as f:
    for epoch in range(NUM_EPOCHS):
        #continue
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
                
            # logits, probas = model(features)
            outputs = model(features)

            ### FORWARD AND BACK PROP
            cost = F.cross_entropy(outputs, targets)
            loss.append(cost)

            optimizer.zero_grad()
            cost.backward()
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
save_checkpoint(checkpoint, filename="resnet50_pretrained_val")
