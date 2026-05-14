import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from tqdm import tqdm

# Handle imports
try:
    from .config import OCRConfig
    from .dataset import create_datasets
except ImportError:
    from config import OCRConfig
    from dataset import create_datasets

# -----------------------------------------------------------------------------
# CRNN Model
# -----------------------------------------------------------------------------
class CRNN(nn.Module):
    def __init__(self, num_classes, img_height=64, hidden_size=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2,1), (2,1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2,1), (2,1)),
        )
        self.cnn_out_height = img_height // 16
        self.rnn = nn.LSTM(512 * self.cnn_out_height, hidden_size, 2,
                           bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w).permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        output = self.classifier(rnn_out)
        return nn.functional.log_softmax(output, dim=2)

# -----------------------------------------------------------------------------
# Collate Function (FIXED)
# -----------------------------------------------------------------------------
def collate_ctc(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)  # Now all images have identical shape (1,64,256)
    # Time steps after CNN: width 256 -> after two MaxPool(2,2): /4 = 64
    time_steps = 64
    input_lengths = torch.full((len(images),), time_steps, dtype=torch.long)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    return images, labels_padded, input_lengths, target_lengths

# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class CTCTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CTCLoss(blank=config.num_classes - 1, zero_infinity=True)
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels, input_lengths, target_lengths in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images).permute(1, 0, 2)  # (T, B, C)
            loss = self.criterion(outputs, labels, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, labels, input_lengths, target_lengths in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images).permute(1, 0, 2)
                loss = self.criterion(outputs, labels, input_lengths, target_lengths)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self):
        best_val_loss = float('inf')
        patience = 0
        print(f"\nTraining on {self.device}, samples: train={len(self.train_loader.dataset)}, val={len(self.val_loader.dataset)}\n")
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step(val_loss)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                torch.save(self.model.state_dict(), os.path.join(self.config.output_dir, 'best_model.pth'))
                print("  -> Best model saved")
            else:
                patience += 1
                if patience >= self.config.early_stopping_patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        self.plot_history()

    def plot_history(self):
        plt.figure()
        plt.plot(self.history['train_loss'], label='Train')
        plt.plot(self.history['val_loss'], label='Val')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(self.config.output_dir, 'loss.png'))
        plt.close()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default='./output')
    args = parser.parse_args()

    config = OCRConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.output_dir = os.path.join(args.output_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    config.use_augmentation = False  # Force disable
    os.makedirs(config.output_dir, exist_ok=True)

    print("Creating datasets...")
    train_dataset, val_dataset, metadata = create_datasets(config)
    config.num_classes = len(metadata['char_to_idx']) + 1

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              sampler=train_dataset.get_weighted_sampler(),
                              num_workers=0, collate_fn=collate_ctc)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=0, collate_fn=collate_ctc)

    model = CRNN(num_classes=config.num_classes, img_height=config.img_height,
                 hidden_size=config.rnn_hidden_size)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    trainer = CTCTrainer(model, train_loader, val_loader, config)
    trainer.train()
    print("Done.")

if __name__ == '__main__':
    main()
