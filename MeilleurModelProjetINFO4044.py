#Code par Alexandre Williams A00211478 Dans le cadre du cours INFO4044 comme projet final

#Ce code est fait pour être exécuté surle serveur Ace-Net
#Changer les chemins d'accès si nécessaire
#Le code doit être sous la forme d'un fichier .py pour être exécuté sur le serveur Ace-Net
#Le code doit être exécuté à travers d'un job
# On doit avoir un fichier job.sh pour avec le code suivant pour lancer le job:

# #!/bin/bash

# #SBATCH --job-name=projet              # Job name
# #SBATCH --nodes=1                      # Number of nodes
# #SBATCH --ntasks=1                     # Number of tasks (processes)
# #SBATCH --cpus-per-task=1              # Number of CPU cores per task
# #SBATCH --mem=8G                       # Memory per node
# #SBATCH --gres=gpu:1                   # Request one GPU
# #SBATCH --time=0-010:00                 # Time limit (days-hours:minutes)

# module load StdEnv/2023
# module load python cuda

# cd /home/$USER/project
# source ./env/bin/activate

# python MeilleurModelProjetINFO4044.py


#Pour lancer le job on doit lancer la commande "sbatch job.sh" dans le terminal


# J'ai aussi un fichier setup.sh pour installer les dépendances nécessaires pour le projet, avant de lancer le job:

# #!/bin/bash

# module load StdEnv/2023
# module load python cuda

# python -m venv env
# source ./env/bin/activate

# pip install --upgrade --no-index pip
# pip install --no-index tensorflow==2.12 keras==2.12
# pip install --no-index jupyter
# pip install --no-index pandas
# pip install --no-index scipy
# pip install --no-index sklearn
# pip install --no-index tf-keras
# pip install --no-index torch
# pip install --no-index torchvision
# pip install --no-index tqdm

#On doit lancer l'exécution du fichier setup.sh avant de lancer le job pour installer les dépendances nécessaires




#Voici les librairies nécessaires pour le code
import pandas as pd
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torch.optim as optim
import pickle

#Faire la lecture des fichier csv besoins
#Changer les chemins d'accès si nécessaire
train_features = pd.read_csv("/home/user023/projet/train_features.csv", index_col="id")
test_features = pd.read_csv("/home/user023/projet/test_features.csv", index_col="id")
train_labels = pd.read_csv("/home/user023/projet/train_labels.csv", index_col="id")

#afficher le nombre de classes
especes = sorted(train_labels.columns.unique())
num_classes = len(especes)
print(f"Nombre de classes: {num_classes}")

#Assurer que la colonne `site` ne se chevauche pas
train_sites, val_sites = train_test_split(
    train_features['site'].unique(), test_size=0.25, random_state=42
)

#Diviser les données par site
x_train = train_features[train_features['site'].isin(train_sites)]
x_val = train_features[train_features['site'].isin(val_sites)]

#Diviser les labels par site
y_train = train_labels.loc[x_train.index]
y_val = train_labels.loc[x_val.index]

#Assurer la distribution des especes
split_pcts = pd.DataFrame(
    {
        "train": y_train.idxmax(axis=1).value_counts(normalize=True),
        "val": y_val.idxmax(axis=1).value_counts(normalize=True),
    }
)
print("Pourcentage d'especes par divisions:")
print((split_pcts.fillna(0) * 100).astype(int))#Afficher les pourcentages des especes par division de train et val

#Vérifier si les sites ne se chevauchent pas
overlap_sites = set(train_sites).intersection(set(val_sites))
if overlap_sites:
    print("Les sites ne sont pas disjoints entre train et val")
else:
    print("Les sites sont disjoints entre train et val")

#Création d'une classe qui hérite de la classe Dataset de Pytorch pour créer un dataset
#Si is_training=True, on applique des transformations pour l'entrainement
#Sinon on applique des transformations pour la validation
class ImagesDataset(Dataset):
    def __init__(self, x_df, y_df=None, is_training=True):
        self.data = x_df
        self.label = y_df
        if is_training:
            #Appliquer les transformations pour l'entrainement
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        else:
            #Application des transformations pour la validation
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

    #Méthode pour obtenir un élément du dataset
    def __getitem__(self, index):
        #On utilise la librairie PIL pour ouvrir l'image et la convertir en RGB
        #On prend la colonne "filepath" pour obtenir le chemin de l'image
        #Les dossiers des images doivent être dans le même dossier que le codepuisque les chemins sont relatifs
        image = Image.open(self.data.iloc[index]["filepath"]).convert("RGB")
        #On applique les transformations sur l'image
        image = self.transform(image)
        #On prend l'index de l'image pour l'identifier
        image_id = self.data.index[index]
        #Si on n'a pas de labels, on retourne l'identifiant de l'image et l'image
        if self.label is None:
            return {"image_id": image_id, "image": image}
        else:
            #Sinon on retourne l'identifiant de l'image, l'image et le label
            #on convertit le label en tensor
            label = torch.tensor(self.label.iloc[index].values, dtype=torch.float)
            return {"image_id": image_id, "image": image, "label": label}
    #Méthode pour obtenir la longueur du dataset
    def __len__(self):
        return len(self.data)


#On crée un dataset pour l'entrainement et un autre pour la validation avec la variable is_training=True pour l'entrainement
train_dataset = ImagesDataset(x_train, y_train, is_training=True)
val_dataset = ImagesDataset(x_val, y_val, is_training=False)

#On crée les dataloaders de la classe DataLoader de Pytorch pour l'entrainement et la validation
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)

#exécuter le code sur un GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Création du modèle avec ResNet50 pré-entrainé sur ImageNet
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
#On change la dernière couche pour définir notre propre classificateur
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(512),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes),
)
#On met le modèle sur le GPU
model = model.to(device)


#On définit la fonction de perte et l'optimiseur avec les paramètres
lossfunc = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)
#On définit le scheduler pour ajuster le taux d'apprentissage comme mentionné dans le rapport
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#Définir le nombre d'époques pour l'entrainement
num_epochs = 20

#Listes pour sauvegarder les métriques
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

best_loss = float("inf")#Variable pour faire comparaison pour sauvegarder le meilleur modèle

#Boucler les époques pour l'entrainement et la validation
for epoch in range(1, num_epochs + 1):
    print(f"epoch {epoch}")
    model.train()#Mettre le modèle en mode entrainement pour activer le dropout
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    #Boucle pour l'entrainement
    #On utilise tqdm pour afficher une barre de progression
    #On boucle sur les batchs du dataloader
    for batch_n, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        optimizer.zero_grad()#Remettre les gradients à zéro après chaque batch
        #On met les données sur le GPU
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        outputs = model(inputs)#Faire la prédiction
        loss = lossfunc(outputs, labels)#Calculer la perte
        loss.backward()#Calculer les gradients
        optimizer.step()#Mettre à jour les poids
        train_loss += loss.item()#Calculer la perte

        #Calculer l'exactitude d'entrainement
        preds = (torch.sigmoid(outputs) > 0.5).int()#Seuil binaire pour les prédictions
        train_correct += (preds == labels.int()).sum().item()#Calculer le nombre de prédictions correctes
        train_total += labels.numel()#Accumuler le nombre total de labels pour calculer l'exactitude

    #Après chaque époques, on sauvegarde la perte et l'exactitude d'entrainement
    train_losses.append(train_loss / len(train_dataloader))#Calculer la perte moyenne
    train_accuracies.append(train_correct / train_total)#Calculer l'exactitude moyenne
    #Afficher les métriques
    print(f"Epoch {epoch} - Training Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.4f}")


    #Boucle pour la validation
    model.eval()#Mettre le modèle en mode évaluation pour désactiver le dropout
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    #Avec torch.no_grad(), on désactive le calcul des gradients pour la validation
    with torch.no_grad():
        #Boucler sur les batchs du dataloader de validation
        for batch in tqdm(val_dataloader, total=len(val_dataloader)):
            #Mettre les données sur le GPU
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)#Faire la prédiction
            loss = lossfunc(outputs, labels)#Calculer la perte
            val_loss += loss.item()#Accumuler la perte

            #Calculer l'exactitude de validation
            preds = (torch.sigmoid(outputs) > 0.5).int()  #Seuil binaire pour les prédictions
            val_correct += (preds == labels.int()).sum().item()#Calculer le nombre de prédictions correctes
            val_total += labels.numel()#Accumuler le nombre total de labels pour calculer l'exactitude

    #Après chaque époques, on sauvegarde la perte et l'exactitude de validation
    val_losses.append(val_loss / len(val_dataloader))#Calculer la perte moyenne et l'ajouter à la liste
    val_accuracies.append(val_correct / val_total)#Calculer l'exactitude moyenne et  l'ajouter à la liste
    #Afficher les métriques
    print(f"Epoch {epoch} - Perte de validation: {val_losses[-1]:.4f}, Exactitude de validation: {val_accuracies[-1]:.4f}")
    #Après chaque époque on regarde si la perte de validation est la plus basse 
    #pour sauvegarder le meilleur modèle
    if val_losses[-1] < best_loss:
        best_loss = val_losses[-1]
        torch.save(model, "/home/user023/projet/models/BestModel.pth")
        print(f"Modèle sauvé avec la plus basse perte de validation: {best_loss:.4f}")
    scheduler.step()#Mettre à jour le scheduler pour ajuster le taux d'apprentissage s'il y a lieu

#Créer un dictionnaire pour sauvegarder l'historique de l'entrainement
history = {}

#Ajouter les exactitudes d'entrainement et de validation à l'historique
history['accuracy'] = train_accuracies
history['val_accuracies'] = val_accuracies
history['loss'] = train_losses
history['val_losses'] = val_losses

#Chemin pour sauvegarder l'historique d'entrainement
file_path = r'/home/user023/projet/history/BestModelHistory.pkl'#Changer le chemin d'accès si nécessaire

#Utiliser pickle pour sauvegarder l'historique
with open(file_path, 'wb') as file:
    pickle.dump(history, file)



#Code pour faire l'inférence sur les données de test et sauvegarder les prédictions dans le format de soumission

#On charge le meilleur modèle sauvegardé lors de l'entrainement
bestModel = torch.load("/home/user023/projet/models/BestModel.pth")#Changer le chemin d'accès si nécessaire
bestModel = bestModel.to(device)

#Créer un dataset pour les données de test
test_dataset = ImagesDataset(test_features.filepath.to_frame(), is_training=False)
test_dataloader = DataLoader(test_dataset, batch_size=32)

#Liste pour sauvegarder les prédictions
preds_collector = []

#Mettre le modèle en mode évaluation
# Liste pour sauvegarder les prédictions
preds_collector = []

# Mettre le modèle en mode évaluation
bestModel.eval()

# Boucle pour l'inférence sur les données de test
with torch.no_grad():
    # Boucler sur les batchs du dataloader de test
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        
        # Transférer les images sur le même appareil que le modèle
        images = batch["image"].to(device)
        
        # Faire la prédiction sur les images
        forward = bestModel.forward(images)
        # Appliquer la fonction softmax pour obtenir les probabilités
        preds = nn.functional.softmax(forward, dim=1)
        # Créer un dataframe avec les prédictions
        preds_df = pd.DataFrame(
            preds.cpu().detach().numpy(),  # Transférer les résultats sur le CPU pour l'encodage
            index=batch["image_id"],
            columns=especes,
        )
        # Ajouter les prédictions à la liste
        preds_collector.append(preds_df)

# Créer la soumission avec les prédictions
submission_df = pd.concat(preds_collector)
# Chargement du format de soumission pour avoir les colonnes et les index pour assurer la bonne forme
submission_format = pd.read_csv(r"/home/user023/projet/submission_format.csv", index_col="id")#Changer le chemin d'accès si nécessaire
# Vérification si les colonnes et les index sont les mêmes
assert all(submission_df.columns == submission_format.columns)
assert all(submission_df.index == submission_format.index)
# Sauvegarder les prédictions dans un fichier csv
submission_df.to_csv('/home/user023/projet/submissionBestModel.csv')#Changer le chemin d'accès si nécessaire
