import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import string
import os
import signal
import sys

# Parâmetros
SEQ_LENGTH = 50
BATCH_SIZE = 128
EPOCHS = 50
LATENT_DIM = 100
HIDDEN_DIM = 512
LEARNING_RATE_G = 0.0001
LEARNING_RATE_D = 0.00005
CHECKPOINT_PATH = 'checkpoint.pth'  # Caminho para salvar o checkpoint

# Preparação do texto
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def clean_text(text):
    text = text.lower()
    allowed_chars = string.ascii_lowercase + string.digits + string.punctuation + ' '
    text = ''.join(c for c in text if c in allowed_chars)
    return text

def text_to_sequences(text, seq_length):
    sequences = []
    next_chars = []
    for i in range(0, len(text) - seq_length):
        sequences.append(text[i:i + seq_length])
        next_chars.append(text[i + seq_length])
    return sequences, next_chars

def create_char_index_dicts(text):
    chars = sorted(list(set(text)))
    char_to_index = {c: i for i, c in enumerate(chars)}
    index_to_char = {i: c for i, c in enumerate(chars)}
    return char_to_index, index_to_char

class TextDataset(Dataset):
    def __init__(self, sequences, next_chars, char_to_index):
        self.sequences = sequences
        self.next_chars = next_chars
        self.char_to_index = char_to_index

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = [self.char_to_index[char] for char in self.sequences[idx]]
        y = self.char_to_index[self.next_chars[idx]]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Definição da GAN
class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, seq_length, vocab_size):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

        self.fc = nn.Linear(latent_dim, hidden_dim * seq_length)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.seq_length, self.hidden_dim)
        lstm_out, _ = self.lstm(x)
        out = self.fc_out(lstm_out)
        return self.softmax(out)

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, seq_length, vocab_size):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        self.lstm = nn.LSTM(vocab_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * seq_length, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.reshape(-1, self.hidden_dim * self.seq_length)
        out = self.fc(lstm_out)
        return self.sigmoid(out)

def save_checkpoint(epoch, step, generator, discriminator, optimizer_G, optimizer_D, loss_G, loss_D, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'loss_G': loss_G,
        'loss_D': loss_D
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}, step {step}")

def load_checkpoint(checkpoint_path, generator, discriminator, optimizer_G, optimizer_D):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']
        loss_G = checkpoint['loss_G']
        loss_D = checkpoint['loss_D']
        print(f"Checkpoint loaded from epoch {start_epoch}, step {start_step}")
        return start_epoch, start_step, loss_G, loss_D
    else:
        print("No checkpoint found, starting training from scratch.")
        return 0, 0, float('inf'), float('inf')

def train_gan(generator, discriminator, data_loader, char_to_index, index_to_char, checkpoint_path):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D)

    start_epoch, start_step, loss_G, loss_D = load_checkpoint(checkpoint_path, generator, discriminator, optimizer_G, optimizer_D)

    def signal_handler(sig, frame):
        print("Interrupt received. Saving checkpoint...")
        save_checkpoint(epoch, step_in_epoch, generator, discriminator, optimizer_G, optimizer_D, g_loss.item(), d_loss.item(), checkpoint_path)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    g_loss = torch.tensor(0.0)
    d_loss = torch.tensor(0.0)

    for epoch in range(start_epoch, EPOCHS):
        step_in_epoch = start_step
        for i, (real_seqs, _) in enumerate(data_loader, start=start_step):
            if step_in_epoch >= 19248:
                break
            batch_size = real_seqs.size(0)

            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Train Discriminator
            z = torch.randn(batch_size, LATENT_DIM)
            fake_seqs = generator(z)

            real_seqs_one_hot = nn.functional.one_hot(real_seqs, num_classes=len(char_to_index)).float()
            fake_seqs = fake_seqs.view(batch_size, SEQ_LENGTH, len(char_to_index))

            discriminator_real = discriminator(real_seqs_one_hot)
            discriminator_fake = discriminator(fake_seqs)

            d_loss_real = criterion(discriminator_real, real_labels)
            d_loss_fake = criterion(discriminator_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # Train Generator (twice as often)
            for _ in range(2):
                z = torch.randn(batch_size, LATENT_DIM)
                fake_seqs = generator(z)

                fake_seqs = fake_seqs.view(batch_size, SEQ_LENGTH, len(char_to_index))
                discriminator_fake = discriminator(fake_seqs)

                g_loss = criterion(discriminator_fake, real_labels)

                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(data_loader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

            step_in_epoch += 1

        save_checkpoint(epoch, 0, generator, discriminator, optimizer_G, optimizer_D, g_loss.item(), d_loss.item(), checkpoint_path)
        start_step = 0  # Reset step after each epoch

def generate_text(generator, index_to_char, seq_length, num_generate=200):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(1, LATENT_DIM)
        generated_seq = generator(z).argmax(dim=-1).view(-1).tolist()
        generated_text = ''.join([index_to_char[idx] for idx in generated_seq])
        return generated_text

def save_generated_text(generated_text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(generated_text)

if __name__ == "__main__":
    input_path = 'C:/Users/SuperBusiness.DESKTOP-V6R5K91/Desktop/CLOD/datasets/textcleaned.txt'
    output_path = 'C:/Users/SuperBusiness.DESKTOP-V6R5K91/Desktop/CLOD/datasets/generated_text.txt'
    
    text = load_text(input_path)
    text = clean_text(text)

    char_to_index, index_to_char = create_char_index_dicts(text)
    sequences, next_chars = text_to_sequences(text, SEQ_LENGTH)

    dataset = TextDataset(sequences, next_chars, char_to_index)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator(LATENT_DIM, HIDDEN_DIM, SEQ_LENGTH, len(char_to_index))
    discriminator = Discriminator(HIDDEN_DIM, SEQ_LENGTH, len(char_to_index))

    print("Starting training...")
    train_gan(generator, discriminator, data_loader, char_to_index, index_to_char, CHECKPOINT_PATH)
    print("Training complete. Generating text...")
    
    generated_text = generate_text(generator, index_to_char, SEQ_LENGTH)
    save_generated_text(generated_text, output_path)
    
    print(f"Generated text saved to {output_path}")    