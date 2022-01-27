from utils import *
from model import *
from trainers import *

def _collate_fn(batch):
    sentences, entities = zip(*batch)
    max_length = max([len(sentence) for sentence in sentences])
    padding_sentences = []
    padding_entities = []
    for sentence, entity in zip(sentences, entities):
        padding_sentences.append(sentence+[0 for j in range(max_length-len(sentence))])
        padding_entities.append(entity+[0 for j in range(max_length-len(sentence))])
    sentences = torch.LongTensor(padding_sentences)
    entities = torch.LongTensor(padding_entities)
    masks = sentences!=0
    return sentences, masks, entities

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, default='word', help='Word or Syllabel data')
parser.add_argument('--data_path', type=str, default='./data/', help='Your data path')
parser.add_argument('--min_freq', type=int, default=1, help='Min frequency of words when create vocab')
parser.add_argument('--save_path', type= str, default='./checkpoint', help='Path to save model')
parser.add_argument('--embedding_size', type= int, default=200, help='Word embedding size')
parser.add_argument('--hidden_size', type= int, default=100, help='Hidden size')
parser.add_argument('--dropout', default=0.01, help='Drop out rate')
parser.add_argument('--num_layers', default=1, help='Number of LSTM layers')
parser.add_argument('--model', type=str, default='BiLSTM', help='Model name')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--device', type=str, default='cpu', help='Running on GPU or CPU')
args = parser.parse_args()

train_sentences, train_tokens = read_data(os.path.join(args.data_path, args.data_type+'/train_'+args.data_type+'.conll'))
val_sentences, val_tokens = read_data(os.path.join(args.data_path, args.data_type+'/dev_'+args.data_type+'.conll'))
test_sentences, test_tokens = read_data(os.path.join(args.data_path, args.data_type+'/test_'+args.data_type+'.conll'))

vocab = Vocab()
vocab.build_vocab(train_sentences, min_freq=args.min_freq)

entity_vocab = EntityVocab()
entity_vocab.build(train_tokens)

train_sentences_encoded, train_entities_encoded = vocab.encode(train_sentences), entity_vocab.encode(train_tokens)
val_sentences_encoded, val_entities_encoded = vocab.encode(val_sentences), entity_vocab.encode(val_tokens)
test_sentences_encoded, test_entities_encoded = vocab.encode(test_sentences), entity_vocab.encode(test_tokens)

trainset = NER_Dataset(train_sentences_encoded, train_entities_encoded)
valset = NER_Dataset(val_sentences_encoded, val_entities_encoded)
testset = NER_Dataset(test_sentences_encoded, test_entities_encoded)

train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=_collate_fn)
val_loader = DataLoader(valset, batch_size=args.batch_size, collate_fn=_collate_fn)
test_loader = DataLoader(testset, batch_size=args.batch_size, collate_fn=_collate_fn)

### Prepare training
if args.model == 'BiLSTM':
    model = NER_BiLSTM(len(vocab), args.embedding_size, args.hidden_size, len(entity_vocab), dropout=args.dropout, n_layers=args.num_layers)
    counter = Counter(token for tokens in train_entities_encoded for token in tokens)
    class_weight = create_class_weight(counter)
    criterion = nn.CrossEntropyLoss(torch.FloatTensor(list(class_weight.values())), reduction='none')
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    trainer = BiLSTM_Trainers(args)
    trainer.train(model, criterion, optimizer, train_loader, val_loader, args.epochs)
elif args.model == 'BiLSTM_CRF':
    model = NER_BiLSTM_CRF(len(vocab), args.embedding_size, args.hidden_size, len(entity_vocab), dropout=args.dropout, n_layers=args.num_layers)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    trainer = BiLSTM_CRF_Trainers(args)
    trainer.train(model, optimizer, train_loader, val_loader, args.epochs)
elif args.model == 'PhoBERT':
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    train_dict = encode_data(tokenizer, train_sentences, entity_vocab.encode(train_tokens))
    val_dict = encode_data(tokenizer, val_sentences, entity_vocab.encode(val_tokens))
    test_dict = encode_data(tokenizer, test_sentences, entity_vocab.encode(test_tokens))
    
    trainset = NER_with_PhoBERT_Dataset(train_dict)
    valset = NER_with_PhoBERT_Dataset(val_dict)
    testset = NER_with_PhoBERT_Dataset(test_dict)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn = _collate_fn)
    val_loader = DataLoader(valset, batch_size=args.batch_size, collate_fn = _collate_fn)
    test_loader = DataLoader(testset, batch_size=args.batch_size, collate_fn = _collate_fn)

    counter = Counter(token for tokens in entity_vocab.encode(train_tokens) for token in tokens)
    class_weight = create_class_weight(counter)
    class_weight[0]=1
    criterion = nn.CrossEntropyLoss(torch.FloatTensor(list(class_weight.values())).to(args.device), reduction='none')
    #criterion = nn.CrossEntropyLoss(reduction='none')
    config = RobertaConfig.from_pretrained('vinai/phobert-base')
    model = PhoBERT_NER(config, len(entity_vocab), args.dropout)
    nn.init.xavier_uniform_(model.classifier.weight)
    model.to(args.device)
    for name, param in model.phobert.named_parameters():  
        param.requires_grad = False
        if "encoder.layer.11" in name or "pooler" in name:
            param.requires_grad = True

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=7e-5, correct_bias=False)

    trainers = PhoBERT_Trainers(args)
    trainers.train(model, criterion, optimizer, train_loader, val_loader, 10)

