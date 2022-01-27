from utils import *


class NER_BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, output_size, dropout=0.1, n_layers=1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size = emb_size, hidden_size = hidden_size, num_layers = n_layers, batch_first = True)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.crf = CRF(output_size, True)
        #self.softmax = nn.LogSoftmax(dim=2)
    def forward(self, data, masks, labels):
        emb = self.embeddings(data)
        output, hidden = self.lstm(emb) 
        output = self.dropout(output)
        output = self.hidden2tag(output)
        output = self.softmax(output)
        return -self.crf(output, labels, masks, 'mean')
    def decode(self, batch, mask):
      emb = self.embeddings(batch)
      out, _ = self.lstm(emb)
      out = self.hidden2tag(out)
      output = self.softmax(out)
      return self.crf.decode(out, mask)

class NER_BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, output_size, dropout=0.1, n_layers=1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size = emb_size, hidden_size = hidden_size//2, num_layers = n_layers, batch_first = True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
    def forward(self, data, masks):
        emb = self.embeddings(data)
        output, hidden = self.lstm(emb) 
        output = self.dropout(output)
        output = self.hidden2tag(output)
        output = self.softmax(output)
        return output.permute(0, 2, 1)

class PhoBERT_NER(RobertaPreTrainedModel):
    def __init__(self, config, output_size, dropout):
        super(PhoBERT_NER, self).__init__(config)
        config.output_hidden_states = True
        self.phobert = RobertaModel.from_pretrained('vinai/phobert-base')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
    def forward(self, input_ids, attention_masks):
        x = self.phobert(input_ids = input_ids, attention_mask = attention_masks)[0]
        x = self.dropout(x)
        x = self.classifier(x)
        return x.permute(0,2,1)