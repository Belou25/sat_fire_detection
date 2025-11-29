import os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
np.set_printoptions(threshold=np.inf)
transform = transforms.ToTensor()

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
def plot_loss_in_episode(episode_number, x, loss) :

    plt.figure(figsize=(13,8))
    plt.plot(x, loss)
    plt.title(f'Loss in Episode {episode_number}')
    plt.xlabel('Image checkpoint')
    plt.ylabel('Dice Loss')
    plt.grid(True)
    plt.savefig(f'figures/in_episode/episode{str(episode_number)}2.png')
    

def plot_loss_out_episode(x, avg_episode_loss, avg_validation_loss) :

    plt.figure(figsize=(13,8))
    plt.plot(x, avg_episode_loss, label = 'avg_episode_loss', color='blue')
    plt.plot(x, avg_validation_loss, label = 'avg_validation_loss', color='green')
    plt.title('Average Training and Validation Loss per Episodes')
    plt.xlabel('Episode numbers')
    plt.ylabel('Dice Loss')
    plt.legend()  
    plt.grid(True)  
    plt.savefig('figures/out_episode/graph2.png')
'''---------------------LOSS FUNCTION---------------------------'''

def dice_loss(pred, target) :
    smooth = 1
    pred = pred.view(-1) #Put the tensor into one vector (no dimensionnality)
    target =  target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. *intersection + smooth)/(pred.sum() + target.sum() + smooth)
    return 1-dice


'''---------------------TRAIN STRAGTEGY OF ONE EPISODE---------------------------'''
def Train_one_episode(model, optimizer, training_data, episode_number) :
    episode_loss = 0
    episode_loss_list = []
    x = []

    print(f'Start training Episode n°{episode_number+1}')
    
    train_loader = DataLoader(training_data, batch_size=8, shuffle=True)

    for i, batch in enumerate(train_loader):
        images, targets = batch #Tuple

        optimizer.zero_grad()

        pred = model(images)
        loss = dice_loss(pred, targets)

        #Backward Propagation and optimization
        loss.backward()
        optimizer.step() #Adjust learning weights

        episode_loss += loss.item() * images.size(0) 
    
        loss = loss.cpu().detach().numpy()

        episode_loss_list.append(loss)
        x.append(i+1)

        #plot_loss_in_episode(episode_number+1, x, episode_loss_list)

        print(f'Batch: {i+1}/343')
        
    return episode_loss/len(train_loader.dataset)
 


'''---------------------MAIN---------------------------'''

Episodes = 20
model = Unet(n_channels=3, n_classes=1)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

Datas = ForestFireDataset()
training_data = Datas.get_list_images_masks(datas_file='FireDataset_40m/train_val_test_data/train.txt')
validation_data = Datas.get_list_images_masks(datas_file='FireDataset_40m/train_val_test_data/val.txt')

best_validation_loss = 1000000
patience = 3 # Number of episode max to wait if the the evaluation score is not betterr than before

x=[]
avg_episode_loss_list = []
avg_validation_loss_list = []

for i in range(Episodes) :
    print(f'Episode n°{i+1}')
    model.train(True)

    avg_episode_loss = Train_one_episode(model, optimizer, training_data, i)
    
    validation_loss =0 
    model.eval()

    with torch.no_grad():
        validation_loader = DataLoader(validation_data, batch_size=8, shuffle=True)
        
        print(f'Start Validation of Episode n°{i+1}')
        for j, vbatch in enumerate(validation_loader) :
            vimages, vtargets = vbatch
            vpred = model(vimages)
            vloss = dice_loss(vpred, vtargets)
            validation_loss += vloss.item() * vimages.size(0)  # On applique un coefficient par nombre d'image 

    avg_validation_loss = validation_loss / len(validation_loader.dataset) # On divise par le nombre d'image totale

    print(f'Episode n°{i+1}, LOSS training : {avg_episode_loss}, LOSS validation : {avg_validation_loss}')
    
    x.append(i+1)
    avg_episode_loss_list.append(avg_episode_loss)
    avg_validation_loss_list.append(avg_validation_loss)    

    plot_loss_out_episode(x, avg_episode_loss_list, avg_validation_loss_list)  

    if avg_validation_loss < best_validation_loss :
        best_validation_loss = avg_validation_loss
        model_path = 'Model_save_weights/model_weights2.pth'
        torch.save(model.state_dict(), model_path)
        episode_no_improve = 0

    else :
        episode_no_improve +=1

    if episode_no_improve>patience :
        print('Early stopping triggered')
        break

   

# Améliorer vitesse by update le plot episode moins souvent.
# 10 episodes =~30 min de train
# Creer le test.py pour test le modele sur les données test et utiliser plusieurs mesures de scores, F1, accuracy ...
# Visualiser les données test pour voir la prédiction 
# La visualisation est la phase la plus importante car une loss de 0.15 = une similarité de 85% entre tout les PIXELS
# -> Si il manque quelques pixels sur la prédiction ce n'est pas très grave²
# Améliorer le score et loss ensuite en tunant les hyperparamètres


'''
def __init__(self):
        self.image_folder_path = 'FireDataset_40m/False_color'
        self.mask_folder_path ='FireDataset_40m/Masks'


    #Put in list all the images names in order in a the file
    def list_all_img(self, path) :
        images = os.listdir(path)
        return images
    
    #get the image with the corrresponding index
    def get_image(self, index) : 
        images = self.list_all_img(self.image_folder_path)
        path_image = os.path.join(self.image_folder_path, '_Sentinel-2 L1C from 2016-08-20_SantaBarbara_BANDS-S2-L1C_0_9.tif')
        self.image = Image.open(path_image)

    #get the mask 
    def get_mask(self, index) : 
        masks = self.list_all_img(self.mask_folder_path)
        path_masks = os.path.join(self.mask_folder_path, '_Sentinel-2 L1C from 2016-08-20_SantaBarbara_BANDS-S2-L1C_0_9.tif')
        self.mask = Image.open(path_masks)

    # Get the matrices 
    def convert_matrices(self) :
        self.get_image(1)
        self.image_array = np.array(self.image)
        self.matrix_r = self.image_array[:, :, 0]
        self.matrix_g = self.image_array[:, :, 1]
        self.matrix_b = self.image_array[:, :, 2]

        self.get_mask(1)
        self.mask_array = np.array(self.mask)

    # Show all the matrices images
    def show_matrices(self):

        self.convert_matrices()
         # Afficher les matrices R, G, B
        fig, axes = plt.subplots(1, 5, figsize=(15, 5))
        
        axes[0].imshow(self.image_array)
        axes[0].set_title('Image')
        axes[0].axis('off')

        axes[1].imshow(self.mask_array, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')

        axes[2].imshow(self.matrix_r, cmap='Reds')
        axes[2].set_title('Matrice R')
        axes[2].axis('off')

        axes[3].imshow(self.matrix_g, cmap='Greens')
        axes[3].set_title('Matrice G')
        axes[3].axis('off')

        axes[4].imshow(self.matrix_b, cmap='Blues')
        axes[4].set_title('Matrice B')
        axes[4].axis('off')

        plt.show()

    # We standarized data between 0 and 1 and put them in en Tensor (to be compatible with Pytorch)
    def standartization(self) :
        self.tensor_r = torch.tensor((self.matrix_r/255.0))
        self.tensor_g = torch.tensor(self.matrix_g/255.0)
        self.tensor_b = torch.tensor(self.matrix_b/255.0)
        self.tensor_mask = torch.tensor(self.mask_array)

'''