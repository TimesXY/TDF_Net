import xlwt
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from utils.TDFNet import TDF_Net
from torchvision import transforms
from torch.utils.data import DataLoader
from process.USDataLoader import USIDatasetFix
from utils.utils_sr import roc_model, confusion, metrics_model

if __name__ == '__main__':
    # Add to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Augmentation
    img_size = 224

    valid_Transform = transforms.Compose([transforms.Resize(img_size),
                                          transforms.CenterCrop(img_size),
                                          transforms.ToTensor()])

    # Load the Dataset
    path = r'D:\MyDataSet\US3M'
    Data_test = USIDatasetFix(path + "\\" + "test.txt", transform=valid_Transform)

    # Delineate the Dataset
    batch_size = 4
    data_test = DataLoader(Data_test, batch_size=batch_size, shuffle=False, pin_memory=True)

    # set the parameters
    epochs = 140

    # build model
    model = TDF_Net(num_class=2, depth=[-5, -4, -3], pretrain=True, channel=512).to(device)

    # load best weight
    model_weight_path = "save_weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # solve the problem of Chinese display messy code
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

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
