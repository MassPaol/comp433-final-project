import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from lightly.loss import NegativeCosineSimilarity
from lightly.transforms import SimSiamTransform
from sklearn.svm import LinearSVC
import torchvision
import random


class SimSiamClassifier:

    def __init__(self,backbone=None,feature_dim=128,epochs=100,batch_size=25,lr=0.05,seed=42,momentum=0.9,weight_decay=5e-4,
    C=1.0,max_iter=5000,device=None):

        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        resnet = torchvision.models.resnet18() # testing a fake backbone
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # testing a fake backbone
        #self.backbone = backbone
        self.simsiam = SimSiamNetwork(backbone=self.backbone,feature_dim=feature_dim).to(self.device)
        self.criterion = NegativeCosineSimilarity()
        self.C = C
        self.max_iter = max_iter
        self.linear_classifier = LinearSVC(C=self.C,max_iter=self.max_iter, random_state=self.seed)
        self.transform = SimSiamTransform(input_size=32)
        self.momentum = momentum
        self.weight_decay = weight_decay

    def prepare_X(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        if X.ndim == 4 and X.shape[-1] == 3:
            X = X.permute(0, 3, 1, 2)
        if X.max() > 1:
            X = X / 255.0
        return X.contiguous()

    class ImageDataset(Dataset):

        def __init__(self, X, transform):
            self.X = X
            self.transform = transform

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            image = self.X[idx]
            views = self.transform(image)

            return views

    def _train_simsiam(self, X_train):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        dataset = self.ImageDataset(X_train,self.transform)
        dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True,drop_last=True,generator=generator)
        optimizer = torch.optim.SGD(self.simsiam.parameters(),lr=self.lr,momentum=self.momentum,weight_decay=self.weight_decay)

        self.simsiam.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            for views in dataloader:
                x0, x1 = views
                x0 = x0.to(self.device)
                x1 = x1.to(self.device)
                z0, p0 = self.simsiam.forward(x0)
                z1, p1 = self.simsiam.forward(x1)
                loss = 0.5 * (self.criterion(z0.detach(), p1)+self.criterion(z1.detach(), p0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()


    def _extract_features(self, X):
        self.backbone.eval()
        features = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = X[start:start + self.batch_size]
                batch = batch.to(self.device)
                f = self.backbone(batch)
                features.append(f.cpu().numpy())

        return np.concatenate(features, axis=0)


    def fit(self, X_train, y_train):

        X_train = self.prepare_X(X_train)
        y_train = np.asarray(y_train)
        self._train_simsiam(X_train)

        for param in self.backbone.parameters():
            param.requires_grad = False

        train_features = self._extract_features(X_train)
        self.linear_classifier.fit(train_features,y_train)

        return self

    def predict(self, X_test):
        X_test = self.prepare_X(X_test)
        test_features = self._extract_features(X_test)
        predictions = self.linear_classifier.predict(test_features)

        return predictions