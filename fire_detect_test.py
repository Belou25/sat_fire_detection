import os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import seaborn as sns
np.set_printoptions(threshold=np.inf)
transform = transforms.ToTensor()

if torch.cuda.is_available() : 
    device = torch.device("cuda")
elif torch.backends.mps.is_available() :
    device = torch.device("mps")
else : 
    device = torch.device("cpu")
print(device)


# torchvision.transforms.ToPILImage(mode=None)
class ForestFireDataset() :
    def __init__(self):
        self.image_folder_path = 'FireDataset_40m/False_color'
        self.mask_folder_path ='FireDataset_40m/Masks'

    '''---------------------OPEN IMAGES AND MASKS---------------------------'''
    
    #Return in a list of tuple image in red, green and blue, mask, and RGB image needed
    def get_list_images_masks(self, datas_file) : 

        #Get all the images names in the file.txt
        with open(datas_file, 'r', encoding='utf-8') as f :
            noms_images = [line.strip() for line in f.readlines()] #Remove\n

        datas = []
        i = 0
        #Get images matrices and apply standardization
        for noms_image in noms_images :
            print(i)
            image_path  = os.path.join(self.image_folder_path, noms_image+str('.tif'))
            mask_path = os.path.join(self.mask_folder_path, noms_image+str('.tif'))
            
            image = Image.open(image_path)
            mask = Image.open(mask_path)
            
            image_tensor = transform(image).to(device) # Transform the image and standarized between [0 and 1]
            '''
            Attention : En transformation Tensor l'image passe de (H*W*C) à (C*H*W) pour etre compatible avec les convolutions (C=3, RGB)
            Il faudra faire un .permute(1,2,0).numpy() pour recup les images
            ou alors torchvision.transforms.ToPILImage(mode=None)
            ''' 
            # Centered the datas around 0 with std(ecart type)=1 for faster learning
            mean_values = torch.mean(image_tensor, dim=[1, 2]).tolist() 
            std_values = torch.std(image_tensor, dim=[1, 2]).tolist()
            std_values = [s if s != 0 else 1.0 for s in std_values]
            normalize = transforms.Normalize(mean=mean_values, std=std_values)

            image_tensor_centered = normalize(image_tensor)
            
            mask_array = np.array(mask)
            mask_tensor = torch.tensor(mask_array, dtype=torch.float32).unsqueeze(0).to(device) #Ajout d'une dimension pour etre compatible (H*W)->(C*H*W)
            datas.append((image_tensor_centered, mask_tensor))
            i+=1
        return datas
    
    def plot_initial(self, datas_file) :
        #Get all the images names in the file.txt
        with open(datas_file, 'r', encoding='utf-8') as f :
            noms_images = [line.strip() for line in f.readlines()] #Remove\n
        
        tuple_array = []
        #Get images matrices and apply standardization
        for noms_image in noms_images :
            image_path  = os.path.join(self.image_folder_path, noms_image+str('.tif'))
            mask_path = os.path.join(self.mask_folder_path, noms_image+str('.tif'))
            
            image = Image.open(image_path)
            mask = Image.open(mask_path)

            image_array = np.array(image)
            matrix_r = image_array[:, :, 0]
            matrix_g = image_array[:, :, 1]
            matrix_b = image_array[:, :, 2]

            mask_array = np.array(mask)
            
            tuple_array.append((image_array, mask_array))

            # Afficher les matrices R, G, B
            fig, axes = plt.subplots(1, 5, figsize=(15, 5))


            axes[0].imshow(image_array)
            axes[0].set_title('Image')
            axes[0].axis('off')

            axes[1].imshow(mask_array, cmap='gray')
            axes[1].set_title('Mask')
            axes[1].axis('off')

            axes[2].imshow(matrix_r, cmap='Reds')
            axes[2].set_title('Matrice R')
            axes[2].axis('off')

            axes[3].imshow(matrix_g, cmap='Greens')
            axes[3].set_title('Matrice G')
            axes[3].axis('off')

            axes[4].imshow(matrix_b, cmap='Blues')
            axes[4].set_title('Matrice B')
            axes[4].axis('off')

            plt.show()

        return tuple_array
    
    '''---------------------U NEURAL NETWORK---------------------------'''

class DoubleConvEncoder(nn.Module) :
    def __init__(self, in_channels, out_channels):
        super(DoubleConvEncoder, self).__init__() # Call all methods of Pytorch NN
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True)
        )

    def forward(self, x) :
        return self.double_conv(x)

class DoubleConvDecoder(nn.Module) :
    def __init__(self, in_channels, out_channels):
        super(DoubleConvDecoder, self).__init__() # Call all methods of Pytorch NN
        
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True))
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same'), #1024 = 512 (upconv) + 512(copy and crop)
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True))
    
    def forward(self, x_upconv, x_skip) : 
        dec = self.upconv(x_upconv)  
        dec = torch.cat((dec, x_skip), dim=1)
        return self.double_conv(dec)
        
        # Attention on a changé Dans MaxPool et Conv Transpose, on a plus de padding=Same ce qui creer l'erreur
        #Pour resoudre crop x_skip pour avoir même taille d'image
class Unet(nn.Module) :
    def __init__(self, n_channels, n_classes):
        super(Unet, self).__init__() # Call all methods of Pytorch NN

        self.n_channels = n_channels
        self.n_classes = n_classes

        #Encoder 
        self.enc1 = DoubleConvEncoder(n_channels, 64)
        self.enc2 = DoubleConvEncoder(64, 128)
        self.enc3 = DoubleConvEncoder(128, 256)
        self.enc4 = DoubleConvEncoder(256, 512)
        
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Baseline
        self.baseline = DoubleConvEncoder(512, 1024)

        #Decoder 
        self.dec4 = DoubleConvDecoder(1024, 512)
        self.dec3 = DoubleConvDecoder(512, 256)
        self.dec2 = DoubleConvDecoder(256, 128)
        self.dec1 = DoubleConvDecoder(128, 64)

        #Exit 
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x) :

        #Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        #Baseline
        baseline = self.baseline(self.pool(enc4))

        #Decoder
        dec4 = self.dec4(baseline, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        return self.final_conv(dec1)

'''--------------------------PLOTS-----------------------------'''

    

def plot_final(tuple_arrays) :

    for tuple in tuple_arrays :

        image_array, mask_array, pred_mask = tuple
        plt.figure(figsize=(13,8))
        
        plt.subplot(2, 2, 1)
        plt.imshow(image_array)
        plt.title('Original Image')

        plt.subplot(2, 2, 2)
        plt.imshow(mask_array, cmap = 'gray')
        plt.title('Label Mask')
        
        plt.subplot(2, 1, 2)
        colored_pred_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        colored_pred_mask[pred_mask == 1] = [0, 255, 0]  # RGBA pour vert vif
        colored_pred_mask[pred_mask == 0] = [0, 0, 0]
        plt.imshow(colored_pred_mask)
        plt.title('Predicted mask')

        plt.tight_layout()

        plt.grid(False)  
        plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
'''---------------------LOSS FUNCTION---------------------------'''

def dice_loss(pred, target) :
    smooth = 1
    pred = pred.view(-1) #Put the tensor into one vector (no dimensionnality)
    target =  target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. *intersection + smooth)/(pred.sum() + target.sum() + smooth)
    return 1-dice

'''---------------------MAIN---------------------------'''

Episodes = 1
model = Unet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load('Model_save_weights/model_weights.pth', weights_only=True, map_location=device))
model.to(device)
model.eval()

Datas = ForestFireDataset()
test_data = Datas.get_list_images_masks(datas_file='FireDataset_40m/train_val_test_data/test.txt')
test2_data = Datas.get_list_images_masks(datas_file='FireDataset_40m/train_val_test_data/test2.txt')
test2_array = Datas.plot_initial(datas_file='FireDataset_40m/train_val_test_data/test2.txt')
validation_loss = 0

for i in range(Episodes) :
    print(f'Episode n°{i+1}')

    all_f1_scores = []
    all_precisions = []
    all_recalls = []
    all_accuracy = []
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        test_loader = DataLoader(test_data, batch_size=8, shuffle=True)
        
        for j, tbatch in enumerate(test_loader) :
            timages, ttargets = tbatch
            vpred = model(timages)
            vloss = dice_loss(vpred, ttargets)
            validation_loss += vloss.item() * timages.size(0)  # On applique un coefficient par nombre d'image 
            # Calcul des métriques
            vpred = vpred.cpu().detach().numpy().squeeze()

            for t in range(len(vpred)) :
                for i in range(len(vpred[0][0])):
                        for k in range(len(vpred[0][0])):
                            if vpred[t][i,k]>=0.01 :
                               vpred[t][i,k] = 1
                            else :
                                vpred[t][i,k] = 0
                            
            f1 = f1_score(ttargets.cpu().numpy().flatten(), vpred.flatten())
            precision = precision_score(ttargets.cpu().numpy().flatten(), vpred.flatten())
            recall = recall_score(ttargets.cpu().detach().numpy().flatten(), vpred.flatten())
            accuracy = accuracy_score(ttargets.cpu().numpy().flatten(), vpred.flatten())

            all_f1_scores.append(f1)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_accuracy.append(accuracy)

            all_y_true.extend(ttargets.cpu().numpy().flatten())
            all_y_pred.extend(vpred.flatten())

    plot_confusion_matrix(all_y_true, all_y_pred)
    avg_validation_loss = validation_loss / len(test_loader.dataset) # On divise par le nombre d'image totale

    # Afficher les métriques moyennes
    print(f'LOSS test : {avg_validation_loss}')
    print(f'Average F1 Score: {np.mean(all_f1_scores)}')
    print(f'Average Precision: {np.mean(all_precisions)}')
    print(f'Average Recall: {np.mean(all_recalls)}')
    print(f'Average Accuracy: {np.mean(all_accuracy)}')


    pred_mask = []
    with torch.no_grad():
            test2_loader = DataLoader(test2_data)

            for j, tbatch in enumerate(test2_loader) :
                timages, ttargets = tbatch
                vpred = model(timages)
                vpred = vpred.cpu().detach().numpy().squeeze()
             
                for i in range(len(vpred)):
                    for k in range(len(vpred)):
                        if vpred[i,k]>=0.01 :
                           vpred[i,k] = 1
                        else :
                            vpred[i,k] = 0
                       
                test2_array[j] = test2_array[j]+(vpred,)
    # Envoyer le tuple test_2_array plus le pred_mask puis les plots sur une même figure. 
    plot_final(test2_array)

    #Ajouter loss pour chaque test 2 