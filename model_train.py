import os
import torch
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt

from utils.TDFNet import TDF_Net
from torchvision import transforms
from utils.utils_train import train
from torch.utils.data import DataLoader
from process.USDataLoader import USIDataset
from utils.utils_sr import roc_model, confusion, metrics_model

if __name__ == '__main__':
    # Add to CUDA
    cache_dir = 'C:/Users/w9629/AppData/Local/Temp/torch/kernels'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cache_dir, exist_ok=True)

    # Data Augmentation
    img_size = 224
    train_Transform = transforms.Compose([transforms.Resize(img_size),
                                          transforms.CenterCrop(img_size),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomVerticalFlip(p=0.5),
                                          transforms.ToTensor()])

    valid_Transform = transforms.Compose([transforms.Resize(img_size),
                                          transforms.CenterCrop(img_size),
                                          transforms.ToTensor()])

    # Load the Dataset
    path = r'D:\MyDataSet\US3M'
    Data_train = USIDataset(path + "\\" + "train.txt", transform=train_Transform)
    Data_valid = USIDataset(path + "\\" + "valid.txt", transform=valid_Transform)
    Data_test = USIDataset(path + "\\" + "test.txt", transform=valid_Transform)

    # Delineate the Dataset
    batch_size = 16
    data_train = DataLoader(Data_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    data_valid = DataLoader(Data_valid, batch_size=batch_size, shuffle=False, pin_memory=True)
    data_test = DataLoader(Data_test, batch_size=batch_size, shuffle=False, pin_memory=True)

    # set the parameters
    epochs = 150
    weight_decay = 1e-5
    learning_rate = 2e-5

    # build model
    model = TDF_Net(num_class=2, depth=[-5, -4, -3], pretrain=True, channel=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # training model
    model, loss = train(model, data_train, data_valid, epochs=epochs, optimizer=optimizer)

    # load best weight
    model_weight_path = "save_weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # solve the problem of Chinese display messy code
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # loss curves
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(range(1, epochs + 1), loss['Loss1'], 'b-', label='train loss')
    ax1.plot(range(1, epochs + 1), loss['Loss2'], 'r-', label='validation loss')
    ax1.set_xlim(1, epochs)
    plt.xlabel("iterations", size=10)
    plt.ylabel("loss", size=10)
    plt.legend(loc=1)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(range(1, epochs + 1), loss['Accuracy1'], 'b-', label='train accuracy')
    ax2.plot(range(1, epochs + 1), loss['Accuracy2'], 'r-', label='validation accuracy')
    ax2.set_ylim(0, 1)
    ax2.set_xlim(1, epochs)
    plt.xlabel("iterations", size=10)
    plt.ylabel("accuracy", size=10)
    plt.legend(loc=1)

    plt.savefig('save_images//TDFN_Loss.jpg', dpi=600)
    plt.show()

    # ROC curves
    fpr_dict, tpr_dict, roc_dict = roc_model(model, data_test, epochs=epochs)

    plt.figure()
    plt.plot(fpr_dict, tpr_dict, label='ROC curve (area = {0:0.4f})'
                                       ''.format(roc_dict), color='r', linestyle='-.', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True  Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('save_images//TDFN_ROC.jpg', dpi=600)
    plt.show()

    # confusion matrix
    cf_matrix = confusion(model, data_test, epochs=epochs)

    plt.figure(figsize=(7, 5))
    ax = sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues')
    ax.title.set_text("Confusion Matrix")
    ax.set_xlabel("Prediction Labels")
    ax.set_ylabel("True Labels")
    plt.savefig('save_images//TDFN_Matrix.jpg', dpi=600)
    plt.show()

    # accuracy of test set
    metrics_model(model, data_test, epochs=epochs)
