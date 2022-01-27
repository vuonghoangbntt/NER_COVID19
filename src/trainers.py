from utils import *
from model import *

class BiLSTM_Trainers:
    def __init__(self, args):
        self.base_name = 'BiLSTM_bs={}_lr={}_maxEpochs={}'.format(args.batch_size, args.learning_rate, args.epochs)
        self.save_path = os.path.join(args.save_path, args.model)
        os.makedirs(self.save_path, mode= 0o666, exist_ok=True)

        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        self.rootLogger = logging.getLogger()
        self.rootLogger.setLevel(logging.INFO)
        reset_logger(self.rootLogger)
        fileHandler = logging.FileHandler("{0}/{1}.log".format(self.save_path, f'experiments_'+self.base_name), mode = 'w')
        fileHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(consoleHandler)
    def train_epoch(self, model, criterion, optimizer, dataset, epoch):
        model.train()
        self.rootLogger.info(f"-----------------------Epoch {epoch}--------------------")
        epoch_loss = []
        for batch in tqdm(dataset, total=len(dataset), desc = 'Train epoch %s'%epoch):
            optimizer.zero_grad()
            data, masks, labels = batch
            out = model(data, masks)
            length = torch.sum(masks)
            loss = torch.sum(criterion(out, labels)*masks)/length
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        return sum(epoch_loss)/len(epoch_loss)
    def eval(self, model, criterion, dataset):
        model.eval()
        val_loss = []
        true_pred =  []
        ground_truth = []
        for batch in tqdm(dataset, total=len(dataset), desc= "Validation"):
            data, masks, labels = batch
            out = model(data, masks)
            length = torch.sum(masks)
            loss = torch.sum(criterion(out, labels)*masks)/length
            val_loss.append(loss.item())
            prediction = torch.argmax(out, dim=1)
            true_pred.append(torch.sum((prediction==labels)*masks))
            ground_truth.append(length)
        return sum(val_loss)/len(val_loss), sum(true_pred)/sum(ground_truth)
    def save_model(self, model, optimizer):
        save_point = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_file = os.path.join(self.save_path, f'{self.base_name}.pt')
        torch.save(save_point, save_file)
    def train(self, model, criterion, optimizer, trainset, valset, epochs):
        self.rootLogger.info("###Start training####")
        best_acc = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(model, criterion, optimizer, trainset, epoch)
            val_loss, val_acc = self.eval(model, criterion, valset)
            self.rootLogger.info(f"Train loss: {train_loss:.3f}\t Val loss: {val_loss:.3f}\t Val acc: {val_acc:.3f}")
            if val_acc>best_acc:
                best_acc = val_acc
                self.save_model(model, optimizer)

class BiLSTM_CRF_Trainers:
    def __init__(self, args):
        self.base_name = 'BiLSTM_CRF_bs={}_lr={}_maxEpochs={}'.format(args.batch_size, args.learning_rate, args.epochs)
        self.save_path = os.path.join(args.save_path, 'BiLSTM_CRF')
        os.makedirs(self.save_path, mode= 0o666, exist_ok=True)

        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        self.rootLogger = logging.getLogger()
        self.rootLogger.setLevel(logging.INFO)
        reset_logger(self.rootLogger)
        fileHandler = logging.FileHandler("{0}/{1}.log".format(self.save_path, f'experiments_'+self.base_name), mode = 'w')
        fileHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(consoleHandler)
    def train_epoch(self, model, optimizer, dataset, epoch):
        model.train()
        self.rootLogger.info(f"-----------------------Epoch {epoch}--------------------")
        epoch_loss = []
        for batch in tqdm(dataset, total=len(dataset), desc = 'Train'):
            optimizer.zero_grad()
            data, masks, labels = batch
            loss = model(data, masks, labels)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        return sum(epoch_loss)/len(epoch_loss)
    def eval(self, model, dataset):
        model.eval()
        val_loss = []
        true_pred =  []
        ground_truth = []
        for batch in tqdm(dataset, total=len(dataset), desc= "Validation"):
            data, masks, labels = batch
            loss = model(data, masks, labels)
            length = torch.sum(masks)
            val_loss.append(loss.item())
            prediction = model.decode(data, None)
            true_pred.append(torch.sum((torch.LongTensor(prediction)==labels)*masks))
            ground_truth.append(length)
        return sum(val_loss)/len(val_loss), sum(true_pred)/sum(ground_truth)
    def save_model(self, model, optimizer):
        save_point = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_file = os.path.join(self.save_path, f'{self.base_name}.pt')
        torch.save(save_point, save_file)
    def train(self, model, optimizer, trainset, valset, epochs):
        self.rootLogger.info("###Start training####")
        best_acc = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(model, optimizer, trainset, epoch)
            val_loss, val_acc = self.eval(model, valset)
            self.rootLogger.info(f"Train loss: {train_loss:.3f}\t Val loss: {val_loss:.3f}\t Val acc: {val_acc:.3f}")
            if val_acc>best_acc:
                best_acc = val_acc
                self.save_model(model, optimizer)

class PhoBERT_Trainers:
    def __init__(self, args):
        self.base_name = 'PhoBERT_bs={}_lr={}_maxEpochs={}'.format(args.batch_size, args.learning_rate, args.epochs)
        self.save_path = os.path.join(args.save_path, 'PhoBERT')
        self.device=  args.device
        os.makedirs(self.save_path, mode= 0o666, exist_ok=True)

        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        self.rootLogger = logging.getLogger()
        self.rootLogger.setLevel(logging.INFO)
        reset_logger(self.rootLogger)
        fileHandler = logging.FileHandler("{0}/{1}.log".format(self.save_path, f'experiments_'+self.base_name), mode = 'w')
        fileHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(consoleHandler)
    def train_epoch(self, model, criterion, optimizer, dataset, epoch):
        model.train()
        self.rootLogger.info(f"-----------------------Epoch {epoch}--------------------")
        epoch_loss = []
        for batch in tqdm(dataset, total=len(dataset), desc=f'Train epoch {epoch}'):
            optimizer.zero_grad()
            input_ids, labels, attention_masks, labels_masks = batch.apply(lambda x: x.to(self.device))
            out = model(input_ids, attention_masks)
            length = torch.sum(labels_masks)
            #print(out.is_cuda, labels.is_cuda, labels_masks.is_cuda, length.is_cuda)
            loss = torch.sum(criterion(out, labels)*labels_masks)/length
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        return sum(epoch_loss)/len(epoch_loss)
    def eval(self, model, criterion, dataset):
        model.eval()
        pred = []
        ground_truth = []
        epoch_loss = []
        for batch in tqdm(dataset, total=len(dataset), desc=f'Evaluation'):
            input_ids, labels, attention_masks, labels_masks = batch.apply(lambda x: x.to(self.device))
            out = model(input_ids, attention_masks)
            predictions = torch.argmax(out, dim=1)
            length = torch.sum(labels_masks)
            loss = torch.sum(criterion(out, labels)*labels_masks)/length
            epoch_loss.append(loss.item())
            for prediction, label in zip(predictions, labels):
                for predict, lab in zip(prediction,label):
                    if lab.item()>=0:
                        pred.append(predict.item())
                        ground_truth.append(lab.item())
                #true_predict.append(torch.sum((predictions==labels)*labels_masks))
                #ground_truth.append(length)
        return sum(epoch_loss)/len(epoch_loss), f1_score(pred, ground_truth, average='macro')
    def save_model(self, model, optimizer):
        save_point = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_file = os.path.join(self.save_path, f'{self.base_name}.pt')
        torch.save(save_point, save_file)
    def train(self, model, criterion, optimizer, trainset, valset, epochs):
        self.rootLogger.info("###Start training####")
        best_acc = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(model, criterion, optimizer, trainset, epoch)
            val_loss, val_acc = eval(model, criterion, valset)
            self.rootLogger.info(f"Train loss: {train_loss:.3f}\t Val loss: {val_loss:.3f}\t Val acc: {val_acc:.3f}")
            if val_acc>best_acc:
                best_acc = val_acc
                self.save_model(model, optimizer)