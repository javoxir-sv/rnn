# -*- coding: utf-8 -*-

import os
import sys
import math
import numpy
import joblib
import random
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ttloop import train, test

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu'
print(f"Current Device: {device}")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

RANDOM_STATE = 42
seed_everything(RANDOM_STATE)

df = pd.read_csv("data/babynames-clean.csv", names=["name", "gender"])
df = df.dropna(subset=['name'])

lens = pd.DataFrame(columns=['length'])
lens['length'] = df['name'].astype(str).str.len()

sns.set_theme(style="whitegrid")
plt.figure(figsize=(6, 4))

max_len = int(lens['length'].max())
min_len = int(lens['length'].min())

sns.histplot(
    data=lens,
    x='length',
    bins=range(min_len, max_len + 2),
    kde=True,
    color='skyblue',
    edgecolor='black'
)

plt.title('Distribution of Baby Name Lengths', fontsize=12, fontweight='bold')
plt.xlabel('Name Length', fontsize=12)
plt.ylabel('Frequency (Count)', fontsize=12)
plt.xticks(range(min_len, max_len + 1))
plt.tight_layout()
plt.show()

print(f"The structural mode (most common length) is: {lens['length'].mode()[0]}")

df_bak = df.copy()

ratio = df['gender'].value_counts()
ratio = ratio.reset_index(level=0)

plt.figure(figsize=(5, 4))
sns.barplot(data=ratio, x='gender', y='count', hue='gender', width=0.5, palette=['#ABC9EA', '#F7C8D0'])
plt.xticks(['boy', 'girl'], [f"boy\n({ratio['count'][0]})", f"girl\n({ratio['count'][1]})"])
plt.title("Baby names gender ratio", fontsize=14)
plt.tight_layout()
plt.show()

letter_count = {key: 0 for key in string.ascii_lowercase}

for idx in range(df.shape[0]):
  name = df['name'][idx].lower()
  for let in name:
    letter_count[let] += 1

letter_count = pd.DataFrame(letter_count, index=[0]).T.reset_index(level=0)

plt.figure(figsize=(6, 5))
sns.barplot(data=letter_count, x='index', y=0, color='#31BCEB')
plt.title("Letter Frequency", fontsize=14)
plt.xlabel('letters')
plt.ylabel('count')
plt.tight_layout()
plt.show()

# Encoding the genders
df['gender'] = df['gender'].apply(lambda g: 1 if g == 'boy' else 0)

# Convert names to lowercase
df['name'] = df['name'].apply(lambda x: x.lower())

# Iterating through all the names in the dataset to find the longest name
max_len = 0
for idx in range(df.shape[0]):
  name = df['name'].iloc[idx]
  if len(name) > max_len:
    print(f"Newest longest name: {name}")
    max_len = len(name)
print(f"Maximum length of a name: {max_len}")

# Adding the prefix, postfix, and the paddings
name_list = []
for idx in range(df.shape[0]):
  name = df['name'][idx].lower()
  lenn = len(name)

  temp_list = []
  temp_list.append("<SOS>")
  for let in name:
    temp_list.append(let)
  temp_list.append("<EOS>")
  for _ in range(max_len-lenn):
    temp_list.append(' ')
  name_list.append(temp_list)

token_to_id = {" " : 0, "<SOS>" : 1, "<EOS>" : 2,}
token_to_id.update({token: id+3 for id, token in enumerate(string.ascii_lowercase)})
id_to_token = {val: key for key, val in token_to_id.items()}

# A small function for conversion
def tokenize(name, token_to_id):
  tokens = []
  for token in name:
    tokens.append(token_to_id[token])
  return tokens

tokenized_names = []
for name in name_list:
  tokenized = tokenize(name, token_to_id)
  tokenized_names.append(tokenized)

X = np.array(tokenized_names)
y = df['gender'].to_numpy()

X_train, X_temp, y_train, y_temp = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, shuffle=True, test_size=0.5, random_state=42)
print(f"Train size: {X_train.shape[0]}\nValid size: {X_valid.shape[0]}\nTest size: {X_test.shape[0]}")

class CDataset(Dataset):
  def __init__(self, X, y, transform=None, target_transform=None):
    self.X = X
    self.y = y
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
    X = self.X[idx]
    y = self.y[idx]
    if self.transform:
        X = self.transform(X)
    if self.target_transform:
        y = self.target_transform(y)
    return X, y

train_dataset = CDataset(X_train, y_train)
valid_dataset = CDataset(X_valid, y_valid)
test_dataset = CDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

class VanillaRNNBlock(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(VanillaRNNBlock, self).__init__()
    self.hidden_size = hidden_size
    self.x2h = nn.Linear(input_size, self.hidden_size, bias=True)
    self.h2h = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    self.tanh = nn.Tanh()

  def forward(self, x_t, h_prev):
    h_next = self.tanh(self.x2h(x_t) + self.h2h(h_prev))
    return h_next

class FullRNN(nn.Module):
  def __init__(self, vocab_size, hidden_size=256, embedding_dim=64):
    super(FullRNN, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
    self.rnn = VanillaRNNBlock(input_size=embedding_dim, hidden_size=self.hidden_size)
    self.char_fc = nn.Linear(self.hidden_size, vocab_size)
    self.gender_fc = nn.Linear(self.hidden_size, 1)
    self.sigmoid = nn.Sigmoid()
    self.dropout = nn.Dropout(0.2)

  # This will initialize the hidden state for the RNN layer
  def init_hidden(self, batch_size, device):
    self.hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
    return self.hidden_state

  def forward(self, x):
    batch_size, seq_len = x.shape
    device = x.device

    embedded = self.embedding(x)
    h_t = self.init_hidden(batch_size, device)
    all_hidden_states = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)

    for t in range(seq_len):
      x_t = embedded[:, t, :]
      h_t = self.rnn(x_t, h_t)
      all_hidden_states[:, t, :] = h_t

    all_hidden_states = self.dropout(all_hidden_states)
    last_hidden = all_hidden_states[:, -1, :]
    char_logits = self.char_fc(all_hidden_states)
    gender_logits = self.gender_fc(last_hidden).squeeze(-1)
    gender_probs = self.sigmoid(gender_logits)

    return char_logits, gender_probs

# Declaring the model, optimizer, loss and the scheduler
rnn = FullRNN(len(id_to_token), hidden_size=512)
rnn = rnn.to(device)

optimizer = optim.Adam(rnn.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.15, last_epoch=-1)
loss_multi = nn.CrossEntropyLoss(ignore_index=0)
loss_bin = nn.BCELoss()

# Saving the losses to analyze them later
train_char_losses = []
train_gender_losses = []
valid_char_losses = []
valid_gender_losses = []

# Training loop, also tests
n_epochs = 10
for epoch in range(n_epochs):
  print(f"Epoch {epoch + 1:02d}:")
  avg_loss_chars_t, avg_loss_genders_t = train(model = rnn, dataloader = train_loader, loss_multi = loss_multi, loss_bin = loss_bin, optimizer=optimizer, device=device, task="char")
  avg_loss_chars_v, avg_loss_genders_v = test(model = rnn, dataloader = valid_loader, loss_multi = loss_multi, loss_bin = loss_bin, device=device)
  scheduler.step()

  train_char_losses.append(avg_loss_chars_t)
  train_gender_losses.append(avg_loss_genders_t)
  valid_char_losses.append(avg_loss_chars_v)
  valid_gender_losses.append(avg_loss_genders_v)

def generate_name(model, id_to_token, token_to_id, prefix=None, temperature=1, max_range=20, device="cpu", retn=False):
  model.eval()
  sos_id = token_to_id.get("<SOS>")
  eos_id = token_to_id.get("<EOS>")
  sequence_ids = [sos_id]

  if prefix:
    for char in prefix:
      char = char.lower()
      sequence_ids.append(token_to_id[char])

  with torch.no_grad():
    for _ in range(max_range - len(sequence_ids)):
      x_tensor = torch.tensor([sequence_ids], dtype=torch.long).to(device)

      char_logits, _ = model(x_tensor)
      last_char_logits = char_logits[0, -1, :]
      last_char_logits = last_char_logits / temperature
      probs = F.softmax(last_char_logits, dim=0)
      next_char_id = torch.multinomial(probs, num_samples=1).item()

      if next_char_id == eos_id:
        break

      sequence_ids.append(next_char_id)

  generated_name = "".join([id_to_token[idx] for idx in sequence_ids[1:]])
  if retn:
    return generated_name
  else:
    print(generated_name)

plt.figure(figsize=(10, 5))
plt.plot(range(1, n_epochs + 1), train_char_losses, label='Train Loss', color='blue', marker='o')
plt.plot(range(1, n_epochs + 1), valid_char_losses, label='Valid Loss', color='orange', marker='s')
plt.title('Character Prediction Loss Convergence')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.grid(True)
plt.legend()
plt.xticks(range(1, n_epochs + 1))
plt.show()

for _ in range(10):
  generate_name(
      model = rnn,
      id_to_token = id_to_token,
      token_to_id = token_to_id,
      temperature=np.random.uniform(0.5, 1.5),
      max_range=random.randrange(4, 14),
      device=device)

test(model = rnn, dataloader = valid_loader, loss_multi = loss_multi, loss_bin = loss_bin, device=device, roc_auc=roc_auc_score)

class GRUBlock(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(GRUBlock, self).__init__()
    self.hidden_size = hidden_size

    self.W_xr = nn.Parameter(torch.empty(input_size, self.hidden_size))
    self.W_xz = nn.Parameter(torch.empty(input_size, self.hidden_size))

    self.W_hr = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
    self.W_hz = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))

    self.b_r = nn.Parameter(torch.empty(1, self.hidden_size))
    self.b_z = nn.Parameter(torch.empty(1, self.hidden_size))

    self.W_xh = nn.Parameter(torch.empty(input_size, self.hidden_size))
    self.W_hh = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
    self.b_h = nn.Parameter(torch.empty(1, self.hidden_size))

    self.init_weights()

  def init_weights(self):
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)

  def forward(self, x_t, h_prev):
    r_t = torch.sigmoid(x_t @ self.W_xr + h_prev @ self.W_hr + self.b_r)
    z_t = torch.sigmoid(x_t @ self.W_xz + h_prev @ self.W_hz + self.b_z)
    h_tilde = torch.tanh(x_t @ self.W_xh + (r_t * h_prev) @ self.W_hh + self.b_h)
    h_next = z_t * h_prev + (1 - z_t) * h_tilde
    return h_next

class FullGRU(nn.Module):
  def __init__(self, vocab_size, hidden_size=256, embedding_dim=64):
    super(FullGRU, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
    self.gru = GRUBlock(input_size=embedding_dim, hidden_size=self.hidden_size)
    self.char_fc = nn.Linear(self.hidden_size, vocab_size)
    self.gender_fc = nn.Linear(self.hidden_size, 1)
    self.sigmoid = nn.Sigmoid()
    self.dropout = nn.Dropout(0.2)

  def init_hidden(self, batch_size, device):
    self.hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
    return self.hidden_state

  def forward(self, x):
    batch_size, seq_len = x.shape
    device = x.device

    embedded = self.embedding(x)
    h_t = self.init_hidden(batch_size, device)
    all_hidden_states = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)

    for t in range(seq_len):
      x_t = embedded[:, t, :]
      h_t = self.gru(x_t, h_t)
      all_hidden_states[:, t, :] = h_t

    all_hidden_states = self.dropout(all_hidden_states)
    last_hidden = all_hidden_states[:, -1, :]
    char_logits = self.char_fc(all_hidden_states)
    gender_logits = self.gender_fc(last_hidden).squeeze(-1)
    gender_probs = self.sigmoid(gender_logits)

    return char_logits, gender_probs

gru = FullGRU(len(id_to_token), hidden_size=512)
gru = gru.to(device)

optimizer = optim.Adam(gru.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.15, last_epoch=-1)
loss_multi = nn.CrossEntropyLoss(ignore_index=0)
loss_bin = nn.BCELoss()

train_char_losses = []
train_gender_losses = []
valid_char_losses = []
valid_gender_losses = []

n_epochs = 10
for epoch in range(n_epochs):
  print(f"Epoch {epoch + 1:02d}:")
  avg_loss_chars_t, avg_loss_genders_t = train(model = gru, dataloader = train_loader, loss_multi = loss_multi, loss_bin = loss_bin, optimizer=optimizer, device=device, task="char")
  avg_loss_chars_v, avg_loss_genders_v = test(model = gru, dataloader = valid_loader, loss_multi = loss_multi, loss_bin = loss_bin, device=device)
  scheduler.step()

  train_char_losses.append(avg_loss_chars_t)
  train_gender_losses.append(avg_loss_genders_t)
  valid_char_losses.append(avg_loss_chars_v)
  valid_gender_losses.append(avg_loss_genders_v)

plt.figure(figsize=(10, 5))
plt.plot(range(1, n_epochs + 1), train_char_losses, label='Train Loss', color='blue', marker='o')
plt.plot(range(1, n_epochs + 1), valid_char_losses, label='Valid Loss', color='orange', marker='s')
plt.title('Character Prediction Loss Convergence')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.grid(True)
plt.legend()
plt.xticks(range(1, n_epochs + 1))
plt.show()

for _ in range(10):
  generate_name(
      model = gru,
      id_to_token = id_to_token,
      token_to_id = token_to_id,
      temperature=np.random.uniform(0.5, 1.5),
      max_range=random.randrange(4, 14),
      device=device)

test(model = gru, dataloader = valid_loader, loss_multi = loss_multi, loss_bin = loss_bin, device=device, roc_auc=roc_auc_score)

class LSTMBlock(nn.Module):
  def __init__(self, input_size, hidden_size, sigma=0.01):
    super(LSTMBlock, self).__init__()
    init_weight = lambda *shape: nn.Parameter(torch.rand(*shape) * sigma)
    triple = lambda: (init_weight(input_size, hidden_size),
                     init_weight(hidden_size, hidden_size),
                     nn.Parameter(torch.zeros(hidden_size)))

    self.W_xi, self.W_hi, self.b_i = triple()
    self.W_xf, self.W_hf, self.b_f = triple()
    self.W_xo, self.W_ho, self.b_o = triple()
    self.W_xc, self.W_hc, self.b_c = triple()

  def forward(self, x_t, states_prev):
    h_prev, c_prev = states_prev

    i_t = torch.sigmoid(x_t @ self.W_xi + h_prev @ self.W_hi + self.b_i)
    f_t = torch.sigmoid(x_t @ self.W_xf + h_prev @ self.W_hf + self.b_f)
    o_t = torch.sigmoid(x_t @ self.W_xo + h_prev @ self.W_ho + self.b_o)
    c_tilde = torch.tanh(x_t @ self.W_xc + h_prev @ self.W_hc + self.b_c)

    c_next = f_t * c_prev + i_t * c_tilde
    h_next = o_t * torch.tanh(c_next)
    return (h_next, c_next)

class FullLSTM(nn.Module):
  def __init__(self, vocab_size, hidden_size=256, embedding_dim=64):
    super(FullLSTM, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
    self.lstm = LSTMBlock(input_size=embedding_dim, hidden_size=self.hidden_size)
    self.char_fc = nn.Linear(self.hidden_size, vocab_size)
    self.gender_fc = nn.Linear(self.hidden_size, 1)
    self.sigmoid = nn.Sigmoid()
    self.dropout = nn.Dropout(0.2)

  # This time LSTM has 2 hidden states, short term memory and long term memory
  def init_hidden(self, batch_size, device):
    h_0 = torch.zeros(batch_size, self.hidden_size, device=device)
    c_0 = torch.zeros(batch_size, self.hidden_size, device=device)
    return (h_0, c_0)

  def forward(self, x):
    batch_size, seq_len = x.shape
    device = x.device

    embedded = self.embedding(x)
    states_t = self.init_hidden(batch_size, device)

    all_hidden_states = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)

    for t in range(seq_len):
      x_t = embedded[:, t, :]
      states_t = self.lstm(x_t, states_t)
      h_t = states_t[0]
      all_hidden_states[:, t, :] = h_t

    all_hidden_states = self.dropout(all_hidden_states)
    last_hidden = all_hidden_states[:, -1, :]
    char_logits = self.char_fc(all_hidden_states)
    gender_logits = self.gender_fc(last_hidden).squeeze(-1)
    gender_probs = self.sigmoid(gender_logits)

    return char_logits, gender_probs

lstm = FullLSTM(len(id_to_token), hidden_size=512)
lstm = lstm.to(device)

optimizer = optim.Adam(lstm.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.15, last_epoch=-1)
loss_multi = nn.CrossEntropyLoss(ignore_index=0)
loss_bin = nn.BCELoss()

train_char_losses = []
train_gender_losses = []
valid_char_losses = []
valid_gender_losses = []

n_epochs = 10
for epoch in range(n_epochs):
  print(f"Epoch {epoch + 1:02d}:")
  avg_loss_chars_t, avg_loss_genders_t = train(model = lstm, dataloader = train_loader, loss_multi = loss_multi, loss_bin = loss_bin, optimizer=optimizer, device=device, task="char")
  avg_loss_chars_v, avg_loss_genders_v = test(model = lstm, dataloader = valid_loader, loss_multi = loss_multi, loss_bin = loss_bin, device=device)
  scheduler.step()

  train_char_losses.append(avg_loss_chars_t)
  train_gender_losses.append(avg_loss_genders_t)
  valid_char_losses.append(avg_loss_chars_v)
  valid_gender_losses.append(avg_loss_genders_v)

plt.figure(figsize=(10, 5))
plt.plot(range(1, n_epochs + 1), train_char_losses, label='Train Loss', color='blue', marker='o')
plt.plot(range(1, n_epochs + 1), valid_char_losses, label='Valid Loss', color='orange', marker='s')
plt.title('Character Prediction Loss Convergence')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.grid(True)
plt.legend()
plt.xticks(range(1, n_epochs + 1))
plt.show()

for _ in range(10):
  generate_name(
      model = lstm,
      id_to_token = id_to_token,
      token_to_id = token_to_id,
      temperature=np.random.uniform(0.5, 1.5),
      max_range=random.randrange(4, 14),
      device=device)

test(model = lstm, dataloader = valid_loader, loss_multi = loss_multi, loss_bin = loss_bin, device=device, roc_auc=roc_auc_score)

# A function to get the raw logits
def get_logits(model, dataloader, device):
  model.eval()
  with torch.no_grad():
    for x_batch, _ in dataloader:
      x_batch = x_batch.to(device)
      char_logits, _ = model(x_batch)
      return char_logits, x_batch

def perplexity_with_ce(logits, targets, pad_idx=0):
  logits_flat = logits.reshape(-1, logits.size(-1))
  targets_flat = targets.reshape(-1)

  ce_loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_idx)
  ppe = torch.exp(ce_loss).item()
  return ppe

def perplexity_without_ce(logits, targets, pad_idx=0):
  probs = F.softmax(logits, dim=-1)

  probs_flat = probs.reshape(-1, logits.size(-1))
  targets_flat = targets.reshape(-1)

  mask = (targets_flat != pad_idx)
  valid_targets = targets_flat[mask]
  valid_probs = probs_flat[mask]

  true_class_probs = valid_probs[torch.arange(len(valid_targets)), valid_targets]
  N = len(valid_targets)
  avg_neg_log_prob = -torch.sum(torch.log(true_class_probs)) / N
  return torch.exp(avg_neg_log_prob).item()

char_logits, x_batch = get_logits(rnn, valid_loader, device)
print(f"with CE: {perplexity_with_ce(char_logits, x_batch)}")
print(f"without CE: {perplexity_without_ce(char_logits, x_batch)}")

models = {"RNN" : FullRNN, "GRU" : FullGRU, "LSTM" : FullLSTM}
random_model  = random.choice(list(models.keys()))
print(f"THE RANDOM MODEL OF THE DAY: {random_model}")

tmodel = models[random_model](len(id_to_token), hidden_size=512)
tmodel.to(device)
char_logits, x_batch = get_logits(tmodel, valid_loader, device)
print(f"with Cross-Entropy: {perplexity_with_ce(char_logits, x_batch)}")
print(f"without Cross-Entropy: {perplexity_without_ce(char_logits, x_batch)}")

# From the previous we still have RNN/GRU/LSTM that are trained models.
char_logits, x_batch = get_logits(lstm, valid_loader, device)
print(f"with Cross-Entropy: {perplexity_with_ce(char_logits, x_batch)}")
print(f"without Cross-Entropy: {perplexity_without_ce(char_logits, x_batch)}")

# A little class to hide print statements.
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

ppe_results = pd.DataFrame(columns=['model', 'with_ce', 'without_ce'])

#rnn
char_logits, x_batch = get_logits(rnn, valid_loader, device)
print("RNN:")
with_ce = perplexity_with_ce(char_logits, x_batch)
without_ce = perplexity_without_ce(char_logits, x_batch)
print(f"with Cross-Entropy: {with_ce}")
print(f"without Cross-Entropy: {without_ce}")
ppe_results.loc[len(ppe_results)] = ['RNN', with_ce, without_ce]

#GRU
char_logits, x_batch = get_logits(gru, valid_loader, device)
print("\nGRU:")
with_ce = perplexity_with_ce(char_logits, x_batch)
without_ce = perplexity_without_ce(char_logits, x_batch)
ppe_results.loc[len(ppe_results)] = ['GRU', with_ce, without_ce]
print(f"with Cross-Entropy: {with_ce}")
print(f"without Cross-Entropy: {without_ce}")

#LSTM
char_logits, x_batch = get_logits(lstm, valid_loader, device)
print("\nLSTM:")
with_ce = perplexity_with_ce(char_logits, x_batch)
without_ce = perplexity_without_ce(char_logits, x_batch)
ppe_results.loc[len(ppe_results)] = ['LSTM', with_ce, without_ce]
print(f"with Cross-Entropy: {with_ce}")
print(f"without Cross-Entropy: {without_ce}")

fig, (ax1, ax2) = plt.subplots(ncols = 2, nrows=1, figsize=(8, 4))
sns.barplot(data=ppe_results, x="model", y='with_ce', ax=ax1, color="#7295F6")
sns.barplot(data=ppe_results, x="model", y='without_ce', ax=ax2, color="#7295F6")
ax1.set_title("with cross-entropy")
ax2.set_title("without cross-entropy")
ax1.set_ylabel("")
ax2.set_ylabel("")
plt.tight_layout()
plt.show()

rnn = FullRNN(len(id_to_token), hidden_size=512).to(device)
gru = FullGRU(len(id_to_token), hidden_size=512).to(device)
lstm = FullLSTM(len(id_to_token), hidden_size=512).to(device)

# Each model has to have different optimizers and schedulers for each optimizer
optimizer_rnn = optim.Adam(rnn.parameters())
optimizer_gru = optim.Adam(gru.parameters())
optimizer_lstm = optim.Adam(lstm.parameters())

scheduler_rnn = optim.lr_scheduler.StepLR(optimizer_rnn, step_size=5, gamma=0.15, last_epoch=-1)
scheduler_gru = optim.lr_scheduler.StepLR(optimizer_gru, step_size=5, gamma=0.15, last_epoch=-1)
scheduler_lstm = optim.lr_scheduler.StepLR(optimizer_lstm, step_size=5, gamma=0.15, last_epoch=-1)

# But we can re-use the same loss function again and again, because it just calculates the error.
loss_multi = nn.CrossEntropyLoss(ignore_index=0)
loss_bin = nn.BCELoss()

res_rnn = pd.DataFrame(columns=["epoch", "model", "Type(Loss/Acc)", "Loss"])
res_gru = pd.DataFrame(columns=["epoch", "model", "Type(Loss/Acc)", "Loss"])
res_lstm = pd.DataFrame(columns=["epoch", "model", "Type(Loss/Acc)", "Loss"])

n_epochs = 10
for epoch in range(n_epochs):
  print(f"Epoch {epoch}:")
  print("RNN:")
  rnn_train = train(model = rnn, dataloader = train_loader, loss_multi = loss_multi, loss_bin = loss_bin, optimizer=optimizer_rnn, device=device, logsoftmax=True, task="gender")
  rnn_test = test(model = rnn, dataloader=valid_loader, loss_multi=loss_multi, loss_bin=loss_bin, device=device, roc_auc=roc_auc_score, return_all=True, logsoftmax=True)
  scheduler_rnn.step()

  print("GRU")
  gru_train = train(model = gru, dataloader = train_loader, loss_multi = loss_multi, loss_bin = loss_bin, optimizer=optimizer_gru, device=device, logsoftmax=True, task="gender")
  gru_test = test(model = gru, dataloader=valid_loader, loss_multi=loss_multi, loss_bin=loss_bin, device=device, roc_auc=roc_auc_score, return_all=True, logsoftmax=True)
  scheduler_gru.step()

  print("LSTM")
  lstm_train = train(model = lstm, dataloader = train_loader, loss_multi = loss_multi, loss_bin = loss_bin, optimizer=optimizer_lstm, device=device, logsoftmax=True, task="gender")
  lstm_test = test(model = lstm, dataloader=valid_loader, loss_multi=loss_multi, loss_bin=loss_bin, device=device, roc_auc=roc_auc_score, return_all=True, logsoftmax=True)
  scheduler_lstm.step()
  print("\n")

  # train() returns = avg_loss_chars, avg_loss_genders
  # test() returns = ((avg_loss_chars, avg_loss_genders), (avg_char_acc, avg_bin_acc), avg_roc)
  res_rnn.loc[len(res_rnn)] = [epoch, "RNN", "train", rnn_train[1]]
  res_rnn.loc[len(res_rnn)] = [epoch, "RNN", "valid", rnn_test[0][1]]

  res_gru.loc[len(res_gru)] = [epoch, "GRU", "train", gru_train[1]]
  res_gru.loc[len(res_gru)] = [epoch, "GRU", "valid", gru_test[0][1]]

  res_lstm.loc[len(res_lstm)] = [epoch, "LSTM", "train", lstm_train[1]]
  res_lstm.loc[len(res_lstm)] = [epoch, "LSTM", "valid", lstm_test[0][1]]

# This time we use seaborn as it's more efficient
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(20, 5.5))
sns.lineplot(data=res_rnn, x="epoch", y="Loss", hue="Type(Loss/Acc)", markers=True, style="Type(Loss/Acc)", legend="brief", ax=axes[0])
sns.lineplot(data=res_gru, x="epoch", y="Loss", hue="Type(Loss/Acc)", markers=True, style="Type(Loss/Acc)", legend="brief", ax=axes[1])
sns.lineplot(data=res_lstm, x="epoch", y="Loss", hue="Type(Loss/Acc)", markers=True, style="Type(Loss/Acc)", legend="brief", ax=axes[2])

for ax, nm in zip(axes, ["RNN", "GRU", "LSTM"]):
  ax.set_title(nm)
  ax.set_ylabel("")
  ax.set_xticks(range(0, n_epochs))

fig.suptitle("Loss Convergence for RNN/GRU/LSTM (binary losses)", y=0.96, fontsize=15)
plt.tight_layout()
plt.show()

genn = pd.DataFrame(columns=['model', 'name'])
for model in [rnn, gru, lstm]:
  md = model.__class__.__name__
  for _ in range(10):
    name = generate_name(
        model = model,
        id_to_token = id_to_token,
        token_to_id = token_to_id,
        temperature=np.random.uniform(0.5, 1.5),
        max_range=random.randrange(4, 14),
        device=device,
        retn=True)
    genn.loc[len(genn)] = [md, name]

genn = genn.assign(row=genn.groupby("model").cumcount()).pivot(index="row", columns="model", values="name").reset_index(drop=True)

for model in [rnn, gru, lstm]:
  print(f"{model.__class__.__name__}:")
  test(model=model, dataloader=test_loader, loss_multi=loss_multi, loss_bin=loss_bin, device=device, roc_auc=roc_auc_score)

mode_length = df['name'].str.len().mode()[0]
df_fixed = df[df['name'].str.len()==mode_length].copy()
max_name_lenght = mode_length+2
print(f"Mode lenth: {mode_length}")

def collect_grads(module, grad_input, grad_output):
  g = grad_output[0].detach()
  x = module.current_input

  for b in range(x.size(0)):
    for t in range(x.size(1)):
      token_idx = x[b, t]
      module.epoch_grads[t, token_idx] += g[b, t]

class DebugRNN(nn.Module):
  def __init__(self, vocab_size, hidden_size=256, embedding_dim=64, device="cpu"):
    super(DebugRNN, self).__init__()
    self.hidden_size = hidden_size

    self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
    self.embedding.epoch_grads = torch.zeros(max_name_lenght, vocab_size, embedding_dim, device=device)
    # This is where we connect the hook:
    self.embedding.register_full_backward_hook(collect_grads)

    self.rnn = VanillaRNNBlock(input_size=embedding_dim, hidden_size=self.hidden_size)
    self.char_fc = nn.Linear(self.hidden_size, vocab_size)
    self.gender_fc = nn.Linear(self.hidden_size, 1)
    self.sigmoid = nn.Sigmoid()
    self.dropout = nn.Dropout(0.2)

  def init_hidden(self, batch_size, device):
    self.hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
    return self.hidden_state

  def forward(self, x):
    batch_size, seq_len = x.shape
    device = x.device

    self.embedding.current_input = x
    embedded = self.embedding(x)

    h_t = self.init_hidden(batch_size, device)
    all_hidden_states = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)
    for t in range(seq_len):
      x_t = embedded[:, t, :]
      h_t = self.rnn(x_t, h_t)
      all_hidden_states[:, t, :] = h_t

    all_hidden_states = self.dropout(all_hidden_states)
    last_hidden = all_hidden_states[:, -1, :]
    char_logits = self.char_fc(all_hidden_states)
    gender_logits = self.gender_fc(last_hidden).squeeze(-1)
    gender_probs = self.sigmoid(gender_logits)
    return char_logits, gender_probs

drnn = DebugRNN(len(id_to_token), hidden_size=512, device=device)
drnn = drnn.to(device)

optimizer = optim.Adam(drnn.parameters())
loss_multi = nn.CrossEntropyLoss(ignore_index=0)
loss_bin = nn.BCELoss()

x_test, y_test = df_fixed.sample(1).iloc[0]

name = x_test.lower()
temp_list = []
temp_list.append("<SOS>")
for let in name:
  temp_list.append(let)
temp_list.append("<EOS>")

x_test = tokenize(temp_list, token_to_id)
x_test, y_test = torch.tensor([x_test]), torch.tensor([y_test], dtype=torch.float32)
x_test, y_test = x_test.to(device), y_test.to(device)

char_logits, gender_probs = drnn(x_test)
loss = loss_bin(gender_probs, y_test)
loss.backward()

grad_norms = torch.norm(drnn.embedding.epoch_grads, dim=-1)
grad_norms_debug = grad_norms.unsqueeze(0)
print(f"{grad_norms_debug.shape}")

for t in range(x_test.size(0)):
  token = x_test[t, 0].item()
  normal_val = grad_norms[t, token].item()
  print(f"Time step: {t}, Token: {token} | Gradient norm: {normal_val:.4f}")
  non_zeros = (grad_norms[0]>0).nonzero(as_tuple=True)[0]
  print(f"Tokens with nonzero gradients: t={t}: {non_zeros.tolist()}")

name_list = []
for idx in range(df_fixed.shape[0]):
  name = df_fixed['name'].iloc[idx].lower()
  lenn = len(name)

  temp_list = []
  temp_list.append("<SOS>")
  for let in name:
    temp_list.append(let)
  temp_list.append("<EOS>")
  name_list.append(temp_list)

tokenized_names = []
for name in name_list:
  tokenized = tokenize(name, token_to_id)
  tokenized_names.append(tokenized)

X_db = np.array(tokenized_names)
y_db = df_fixed['gender'].to_numpy()

X_train_db, X_valid_db, y_train_db, y_valid_db = train_test_split(X_db, y_db, random_state=42, shuffle=True, test_size=0.2)

train_dataset_db = CDataset(X_train_db, y_train_db)
valid_dataset_db = CDataset(X_valid_db, y_valid_db)

train_loader_db = DataLoader(train_dataset_db, batch_size=32, shuffle=True)
valid_loader_db = DataLoader(valid_dataset_db, batch_size=32, shuffle=True)

# Hooking up the other 2 models as well.
class DebugGRU(nn.Module):
  def __init__(self, vocab_size, hidden_size=256, embedding_dim=64, device="cpu"):
    super(DebugGRU, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
    self.embedding.epoch_grads = torch.zeros(max_name_lenght, vocab_size, embedding_dim, device=device)
    self.embedding.register_full_backward_hook(collect_grads)
    self.gru = GRUBlock(input_size=embedding_dim, hidden_size=self.hidden_size)
    self.char_fc = nn.Linear(self.hidden_size, vocab_size)
    self.gender_fc = nn.Linear(self.hidden_size, 1)
    self.sigmoid = nn.Sigmoid()
    self.dropout = nn.Dropout(0.2)

  def init_hidden(self, batch_size, device):
    self.hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
    return self.hidden_state

  def forward(self, x):
    batch_size, seq_len = x.shape
    device = x.device
    self.embedding.current_input = x
    embedded = self.embedding(x)
    h_t = self.init_hidden(batch_size, device)
    all_hidden_states = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)

    for t in range(seq_len):
      x_t = embedded[:, t, :]
      h_t = self.gru(x_t, h_t)
      all_hidden_states[:, t, :] = h_t

    all_hidden_states = self.dropout(all_hidden_states)
    last_hidden = all_hidden_states[:, -1, :]
    char_logits = self.char_fc(all_hidden_states)
    gender_logits = self.gender_fc(last_hidden).squeeze(-1)
    gender_probs = self.sigmoid(gender_logits)
    return char_logits, gender_probs

class DebugLSTM(nn.Module):
  def __init__(self, vocab_size, hidden_size=256, embedding_dim=64, device="cpu"):
    super(DebugLSTM, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
    self.embedding.epoch_grads = torch.zeros(max_name_lenght, vocab_size, embedding_dim, device=device)
    self.embedding.register_full_backward_hook(collect_grads)
    self.lstm = LSTMBlock(input_size=embedding_dim, hidden_size=self.hidden_size)
    self.char_fc = nn.Linear(self.hidden_size, vocab_size)
    self.gender_fc = nn.Linear(self.hidden_size, 1)
    self.sigmoid = nn.Sigmoid()
    self.dropout = nn.Dropout(0.2)

  def init_hidden(self, batch_size, device):
    h_0 = torch.zeros(batch_size, self.hidden_size, device=device)
    c_0 = torch.zeros(batch_size, self.hidden_size, device=device)
    return (h_0, c_0)

  def forward(self, x):
    batch_size, seq_len = x.shape
    device = x.device
    self.embedding.current_input = x
    embedded = self.embedding(x)
    states_t = self.init_hidden(batch_size, device)
    all_hidden_states = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)
    for t in range(seq_len):
      x_t = embedded[:, t, :]
      states_t = self.lstm(x_t, states_t)
      h_t = states_t[0]
      all_hidden_states[:, t, :] = h_t

    all_hidden_states = self.dropout(all_hidden_states)
    last_hidden = all_hidden_states[:, -1, :]
    char_logits = self.char_fc(all_hidden_states)
    gender_logits = self.gender_fc(last_hidden).squeeze(-1)
    gender_probs = self.sigmoid(gender_logits)
    return char_logits, gender_probs

drnn = DebugRNN(len(id_to_token), hidden_size=512, device=device)
dgru = DebugGRU(len(id_to_token), hidden_size=512, device=device)
dlstm = DebugLSTM(len(id_to_token), hidden_size=512, device=device)
drnn = drnn.to(device)
dgru = dgru.to(device)
dlstm = dlstm.to(device)

optimizer_drnn = optim.Adam(drnn.parameters())
optimizer_dgru = optim.Adam(dgru.parameters())
optimizer_dlstm = optim.Adam(dlstm.parameters())

scheduler_rnn = optim.lr_scheduler.StepLR(optimizer_rnn, step_size=5, gamma=0.15, last_epoch=-1)
scheduler_gru = optim.lr_scheduler.StepLR(optimizer_gru, step_size=5, gamma=0.15, last_epoch=-1)
scheduler_lstm = optim.lr_scheduler.StepLR(optimizer_lstm, step_size=5, gamma=0.15, last_epoch=-1)

loss_multi = nn.CrossEntropyLoss(ignore_index=0)
loss_bin = nn.BCELoss()

n_epochs = 10
all_epochs_grads_rnn = torch.zeros(n_epochs, max_name_lenght, len(id_to_token), 64)
all_epochs_grads_gru = torch.zeros(n_epochs, max_name_lenght, len(id_to_token), 64)
all_epochs_grads_lstm = torch.zeros(n_epochs, max_name_lenght, len(id_to_token), 64)

for epoch in range(n_epochs):
  print(f"Epoch {epoch + 1:02d}...")
  with HiddenPrints():
    drnn.embedding.epoch_grads.zero_()
    dgru.embedding.epoch_grads.zero_()
    dlstm.embedding.epoch_grads.zero_()

    train(model=drnn, dataloader=train_loader_db, loss_multi=loss_multi, loss_bin=loss_bin, optimizer=optimizer_drnn, device=device, task="gender")
    train(model=dgru, dataloader=train_loader_db, loss_multi=loss_multi, loss_bin=loss_bin, optimizer=optimizer_dgru, device=device, task="gender")
    train(model=dlstm, dataloader=train_loader_db, loss_multi=loss_multi, loss_bin=loss_bin, optimizer=optimizer_dlstm, device=device, task="gender")

    scheduler_rnn.step()
    scheduler_gru.step()
    scheduler_lstm.step()

    all_epochs_grads_rnn[epoch] = drnn.embedding.epoch_grads.clone()
    all_epochs_grads_gru[epoch] = dgru.embedding.epoch_grads.clone()
    all_epochs_grads_lstm[epoch] = dlstm.embedding.epoch_grads.clone()

print("Finished.")

frobenius_norm_rnn = torch.linalg.norm(all_epochs_grads_rnn, dim=(2, 3), ord="fro")
frobenius_mean_rnn = frobenius_norm_rnn.mean(dim=0)

frobenius_norm_gru = torch.linalg.norm(all_epochs_grads_gru, dim=(2, 3), ord="fro")
frobenius_mean_gru = frobenius_norm_gru.mean(dim=0)

frobenius_norm_lstm = torch.linalg.norm(all_epochs_grads_lstm, dim=(2, 3), ord="fro")
frobenius_mean_lstm = frobenius_norm_lstm.mean(dim=0)

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(16, 4))
sns.lineplot(x=range(0, max_name_lenght), y=frobenius_mean_rnn, ax=axes[0])
sns.lineplot(x=range(0, max_name_lenght), y=frobenius_mean_gru, ax=axes[1])
sns.lineplot(x=range(0, max_name_lenght), y=frobenius_mean_lstm, ax=axes[2])

for ax, nm in zip(axes, ["RNN", "GRU", "LSTM"]):
  ax.set_title(nm)
  ax.set_ylabel("")
  ax.set_ylim(bottom=0)
  ax.set_xticks(range(0, max_name_lenght))

fig.suptitle("Gradient Curves for binary classification", y=0.96, fontsize=15)
plt.tight_layout()
plt.show()

# Resetting the model weights
for model in [drnn, dgru, dlstm]:
  for layer in model.children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

# As well as the optimizers
optimizer_drnn = optim.Adam(drnn.parameters())
optimizer_dgru = optim.Adam(dgru.parameters())
optimizer_dlstm = optim.Adam(dlstm.parameters())

scheduler_rnn = optim.lr_scheduler.StepLR(optimizer_rnn, step_size=5, gamma=0.15, last_epoch=-1)
scheduler_gru = optim.lr_scheduler.StepLR(optimizer_gru, step_size=5, gamma=0.15, last_epoch=-1)
scheduler_lstm = optim.lr_scheduler.StepLR(optimizer_lstm, step_size=5, gamma=0.15, last_epoch=-1)

n_epochs = 10
all_epochs_grads_rnn = torch.zeros(n_epochs, max_name_lenght, len(id_to_token), 64)
all_epochs_grads_gru = torch.zeros(n_epochs, max_name_lenght, len(id_to_token), 64)
all_epochs_grads_lstm = torch.zeros(n_epochs, max_name_lenght, len(id_to_token), 64)

for epoch in range(n_epochs):
  with HiddenPrints():
    drnn.embedding.epoch_grads.zero_()
    dgru.embedding.epoch_grads.zero_()
    dlstm.embedding.epoch_grads.zero_()

    train(model=drnn, dataloader=train_loader_db, loss_multi=loss_multi, loss_bin=loss_bin, optimizer=optimizer_drnn, device=device, task="char")
    train(model=dgru, dataloader=train_loader_db, loss_multi=loss_multi, loss_bin=loss_bin, optimizer=optimizer_dgru, device=device, task="char")
    train(model=dlstm, dataloader=train_loader_db, loss_multi=loss_multi, loss_bin=loss_bin, optimizer=optimizer_dlstm, device=device, task="char")

    scheduler_rnn.step()
    scheduler_gru.step()
    scheduler_lstm.step()

    all_epochs_grads_rnn[epoch] = drnn.embedding.epoch_grads.clone()
    all_epochs_grads_gru[epoch] = dgru.embedding.epoch_grads.clone()
    all_epochs_grads_lstm[epoch] = dlstm.embedding.epoch_grads.clone()
print("Training finished..")

frobenius_norm_rnn = torch.linalg.norm(all_epochs_grads_rnn, dim=(2, 3), ord="fro")
frobenius_mean_rnn = frobenius_norm_rnn.mean(dim=0)

frobenius_norm_gru = torch.linalg.norm(all_epochs_grads_gru, dim=(2, 3), ord="fro")
frobenius_mean_gru = frobenius_norm_gru.mean(dim=0)

frobenius_norm_lstm = torch.linalg.norm(all_epochs_grads_lstm, dim=(2, 3), ord="fro")
frobenius_mean_lstm = frobenius_norm_lstm.mean(dim=0)

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(16, 4))
sns.lineplot(x=range(0, max_name_lenght), y=frobenius_mean_rnn, ax=axes[0])
sns.lineplot(x=range(0, max_name_lenght), y=frobenius_mean_gru, ax=axes[1])
sns.lineplot(x=range(0, max_name_lenght), y=frobenius_mean_lstm, ax=axes[2])

for ax, nm in zip(axes, ["RNN", "GRU", "LSTM"]):
  ax.set_title(nm)
  ax.set_ylabel("")
  ax.set_ylim(bottom=0)
  ax.set_xticks(range(0, max_name_lenght))

fig.suptitle("Gradient Curves for multi-class classification", y=0.96, fontsize=15)
plt.tight_layout()
plt.show()

rnn = FullRNN(len(id_to_token), hidden_size=512)
gru = FullGRU(len(id_to_token), hidden_size=512)
lstm = FullLSTM(len(id_to_token), hidden_size=512)
rnn = rnn.to(device)
gru = gru.to(device)
lstm = lstm.to(device)

optimizer_rnn = optim.Adam(rnn.parameters())
optimizer_gru = optim.Adam(gru.parameters())
optimizer_lstm = optim.Adam(lstm.parameters())

scheduler_rnn = optim.lr_scheduler.StepLR(optimizer_rnn, step_size=5, gamma=0.15, last_epoch=-1)
scheduler_gru = optim.lr_scheduler.StepLR(optimizer_gru, step_size=5, gamma=0.15, last_epoch=-1)
scheduler_lstm = optim.lr_scheduler.StepLR(optimizer_lstm, step_size=5, gamma=0.15, last_epoch=-1)

# The loss functions
loss_multi = nn.CrossEntropyLoss(ignore_index=0)
loss_bin = nn.BCELoss()

res_rnn = pd.DataFrame(columns=["epoch", "model", "dataset", "Loss(multi)", "Loss(binary)"])
res_gru = pd.DataFrame(columns=["epoch", "model", "dataset", "Loss(multi)", "Loss(binary)"])
res_lstm = pd.DataFrame(columns=["epoch", "model", "dataset", "Loss(multi)", "Loss(binary)"])

n_epochs = 10
for epoch in range(n_epochs):
  print(f"Epoch {epoch+1:02d}:")
  with HiddenPrints():
    rnn_train = train(model=rnn, dataloader=train_loader, loss_multi=loss_multi, loss_bin=loss_bin, optimizer=optimizer_rnn, device=device, task="both")
    rnn_test = test(model=rnn, dataloader=valid_loader, loss_multi=loss_multi, loss_bin=loss_bin, device=device)
    scheduler_rnn.step()

    gru_train = train(model=gru, dataloader=train_loader, loss_multi=loss_multi, loss_bin=loss_bin, optimizer=optimizer_gru, device=device, task="both")
    gru_test = test(model=gru, dataloader=valid_loader, loss_multi=loss_multi, loss_bin=loss_bin, device=device)
    scheduler_gru.step()

    lstm_train = train(model=lstm, dataloader=train_loader, loss_multi=loss_multi, loss_bin = loss_bin, optimizer=optimizer_lstm, device=device, task="both")
    lstm_test = test(model=lstm, dataloader=valid_loader, loss_multi=loss_multi, loss_bin=loss_bin, device=device)
    scheduler_lstm.step()

  # train() => avg_loss_chars, avg_loss_genders
  # test() => avg_loss_chars, avg_loss_genders
  res_rnn.loc[len(res_rnn)] = [epoch+1, "RNN", "train", rnn_train[0], rnn_train[1]]
  res_rnn.loc[len(res_rnn)] = [epoch+1, "RNN", "valid", rnn_test[0], rnn_test[1]]

  res_gru.loc[len(res_gru)] = [epoch+1, "GRU", "train", gru_train[0], gru_train[1]]
  res_gru.loc[len(res_gru)] = [epoch+1, "GRU", "valid", gru_test[0], gru_test[1]]

  res_lstm.loc[len(res_lstm)] = [epoch+1, "LSTM", "train", lstm_train[0], lstm_train[1]]
  res_lstm.loc[len(res_lstm)] = [epoch+1, "LSTM", "valid", lstm_test[0], lstm_test[1]]

print("Finished.")

fig, axesm = plt.subplots(ncols=3, nrows=1, figsize=(16, 4.2))
sns.lineplot(data=res_rnn, x="epoch", y="Loss(multi)", hue="dataset", markers=True, style="dataset", legend="brief", ax=axesm[0])
sns.lineplot(data=res_gru, x="epoch", y="Loss(multi)", hue="dataset", markers=True, style="dataset", legend="brief", ax=axesm[1])
sns.lineplot(data=res_lstm, x="epoch", y="Loss(multi)", hue="dataset", markers=True, style="dataset", legend="brief", ax=axesm[2])
fig.suptitle("Loss Convergence for RNN/GRU/LSTM (next letter)", y=0.96, fontsize=15)

for ax, nm in zip(axesm, ["RNN", "GRU", "LSTM"]):
  ax.set_title(nm)
  ax.set_ylabel("")
  ax.set_xticks(range(0, n_epochs))

plt.tight_layout()

fig, axesb = plt.subplots(ncols=3, nrows=1, figsize=(16, 4.2))
sns.lineplot(data=res_rnn, x="epoch", y="Loss(binary)", hue="dataset", markers=True, style="dataset", legend="brief", ax=axesb[0])
sns.lineplot(data=res_gru, x="epoch", y="Loss(binary)", hue="dataset", markers=True, style="dataset", legend="brief", ax=axesb[1])
sns.lineplot(data=res_lstm, x="epoch", y="Loss(binary)", hue="dataset", markers=True, style="dataset", legend="brief", ax=axesb[2])
fig.suptitle("Loss Convergence for RNN/GRU/LSTM (gender)", y=0.96, fontsize=15)

for ax, nm in zip(axesb, ["RNN", "GRU", "LSTM"]):
  ax.set_title(nm)
  ax.set_ylabel("")
  ax.set_xticks(range(0, n_epochs))

plt.tight_layout()
plt.show()

genn = pd.DataFrame(columns=['model', 'name'])
for model in [rnn, gru, lstm]:
  md = model.__class__.__name__
  for _ in range(10):
    name = generate_name(
        model = model,
        id_to_token = id_to_token,
        token_to_id = token_to_id,
        temperature=np.random.uniform(0.5, 1.5),
        max_range=random.randrange(4, 14),
        device=device,
        retn=True)
    genn.loc[len(genn)] = [md, name]

genn = genn.assign(row=genn.groupby("model").cumcount()).pivot(index="row", columns="model", values="name").reset_index(drop=True)

for model in [rnn, gru, lstm]:
  print(f"{model.__class__.__name__}:")
  test(model=model, dataloader=test_loader, loss_multi=loss_multi, loss_bin=loss_bin, device=device, roc_auc=roc_auc_score)

embeddings_weight = lstm.embedding.weight.detach().cpu().numpy()
tokens = [id_to_token[i] for i in range(len(id_to_token))]

special_tokens = {'<SOS>', '<EOS>', '0', '1', '2'}
vowels = {'a', 'e', 'i', 'o', 'u', 'y'}

filtered_indices = [i for i, t in enumerate(tokens) if t not in special_tokens]
filtered_tokens = [tokens[i] for i in filtered_indices]
filtered_embeddings = embeddings_weight[filtered_indices]

categories = []
for token in filtered_tokens:
    if token.lower() in vowels:
        categories.append('Vowel')
    else:
        categories.append('Consonant')

df_emb = pd.DataFrame({
    'token': filtered_tokens,
    'category': categories
})

tsne = TSNE(n_components=2, perplexity=5, random_state=42, max_iter=1000, learning_rate='auto')
tsne_results = tsne.fit_transform(filtered_embeddings)
df_emb['tSNE_1'] = tsne_results[:, 0]
df_emb['tSNE_2'] = tsne_results[:, 1]

pca = PCA(n_components=2, random_state=42)
pca_results = pca.fit_transform(filtered_embeddings)
df_emb['PCA_1'] = pca_results[:, 0]
df_emb['PCA_2'] = pca_results[:, 1]

sns.set_theme(style="whitegrid")
palette = {'Vowel': '#f43f5e', 'Consonant': '#0ea5e9'}
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA
sns.scatterplot(
    data=df_emb, x='PCA_1', y='PCA_2', hue='category',
    palette=palette, s=150, alpha=0.8, ax=axes[0], edgecolor='w', linewidth=1.5
)
axes[0].set_title("PCA Letter Embeddings (Linear)", fontsize=14, pad=15)

# t-SNE
sns.scatterplot(
    data=df_emb, x='tSNE_1', y='tSNE_2', hue='category',
    palette=palette, s=150, alpha=0.8, ax=axes[1], edgecolor='w', linewidth=1.5
)
axes[1].set_title("t-SNE Letter Embeddings (Non-Linear)", fontsize=14, pad=15)

for i, row in df_emb.iterrows():
    axes[0].annotate(
        row['token'], (row['PCA_1'], row['PCA_2']),
        textcoords="offset points", xytext=(0,10), ha='center',
        fontsize=11, fontweight='bold', color='#1e293b')

    axes[1].annotate(
        row['token'], (row['tSNE_1'], row['tSNE_2']),
        textcoords="offset points", xytext=(0,10), ha='center',
        fontsize=11, fontweight='bold', color='#1e293b')

plt.tight_layout()
plt.show()

joblib.dump(lstm, 'data/lstm.pkl')
