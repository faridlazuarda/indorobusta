def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def count_param(module, trainable=False):
    if trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)

def init_model(id_model):
    if id_model == 1:
        tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
        config = BertConfig.from_pretrained('indobenchmark/indobert-base-p2')
        config.num_labels = DocumentSentimentDataset.NUM_LABELS
        model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2', config=config)
    elif id_model == 2:
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        config = XLMRobertaConfig.from_pretrained("xlm-roberta-base")
        config.num_labels = DocumentSentimentDataset.NUM_LABELS
        model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', config=config)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        config = BertConfig.from_pretrained("bert-base-multilingual-uncased")
        config.num_labels = DocumentSentimentDataset.NUM_LABELS
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', config=config)

    return tokenizer, config, model

def load_dataset_loader(dataset_id, ds_type, tokenizer):
    dataset_path = None
    dataset = None
    loader = None
    if(dataset_id == 'sentiment'):
        if(ds_type == "train"):
            dataset_path = '../input/smsa-docsentimentprosa/dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv'
            dataset = DocumentSentimentDataset(dataset_path, tokenizer, lowercase=True)
            loader = DocumentSentimentDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=16, shuffle=True)  
        elif(ds_type == "valid"):
            dataset_path = '../input/smsa-docsentimentprosa/dataset/smsa_doc-sentiment-prosa/valid_preprocess.tsv'
            dataset = DocumentSentimentDataset(dataset_path, tokenizer, lowercase=True)
            loader = DocumentSentimentDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=16, shuffle=False)
        elif(ds_type == "test"):
            dataset_path = '../input/smsa-docsentimentprosa/dataset/smsa_doc-sentiment-prosa/test_preprocess_masked_label.tsv'
            dataset = DocumentSentimentDataset(dataset_path, tokenizer, lowercase=True)
            loader = DocumentSentimentDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=16, shuffle=False)

    elif(dataset_id == 'emotion'):
        if(ds_type == "train"):
            dataset_path = '../input/emot-emotiontwitter/train_preprocess.csv'
            dataset = EmotionDetectionDataset(dataset_path, tokenizer, lowercase=True)
            loader = EmotionDetectionDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=16, shuffle=True)  
        elif(ds_type == "valid"):
            dataset_path = '../input/emot-emotiontwitter/train_preprocess.csv'
            dataset = EmotionDetectionDataset(dataset_path, tokenizer, lowercase=True)
            loader = EmotionDetectionDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=16, shuffle=False)
        elif(ds_type == "test"):
            dataset_path = '../input/emot-emotiontwitter/valid_preprocess.csv'
            dataset = EmotionDetectionDataset(dataset_path, tokenizer, lowercase=True)
            loader = EmotionDetectionDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=16, shuffle=False)

    return dataset, loader

def text_logit(text, model, tokenizer, i2w):
    subwords = tokenizer.encode(text)
    subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)
    
    logits = model(subwords)[0]
    label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

    print(f'Text: {text} | Label : {i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)')

def fine_tuning_model(base_model, i2w, train_loader, valid_loader, epochs=5):
    optimizer = optim.Adam(base_model.parameters(), lr=3e-6)
    base_model = base_model.cuda()
    
    # Train
    n_epochs = epochs
    for epoch in range(n_epochs):
        base_model.train()
        torch.set_grad_enabled(True)

        total_train_loss = 0
        list_hyp, list_label = [], []

        train_pbar = tqdm(train_loader, leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            # Forward base_model
            loss, batch_hyp, batch_label = forward_sequence_classification(base_model, batch_data[:-1], i2w=i2w, device='cuda')

            # Update base_model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss = loss.item()
            total_train_loss = total_train_loss + tr_loss

            # Calculate metrics
            list_hyp += batch_hyp
            list_label += batch_label

            train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                total_train_loss/(i+1), get_lr(optimizer)))

        # Calculate train metric
        metrics = document_sentiment_metrics_fn(list_hyp, list_label)
        print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
            total_train_loss/(i+1), metrics_to_string(metrics), get_lr(optimizer)))

        # Evaluate on validation
        base_model.eval()
        torch.set_grad_enabled(False)

        total_loss, total_correct, total_labels = 0, 0, 0
        list_hyp, list_label = [], []

        pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
        for i, batch_data in enumerate(pbar):
            batch_seq = batch_data[-1]        
            loss, batch_hyp, batch_label = forward_sequence_classification(base_model, batch_data[:-1], i2w=i2w, device='cuda')

            # Calculate total loss
            valid_loss = loss.item()
            total_loss = total_loss + valid_loss

            # Calculate evaluation metrics
            list_hyp += batch_hyp
            list_label += batch_label
            metrics = document_sentiment_metrics_fn(list_hyp, list_label)

            pbar.set_description("VALID LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))

        metrics = document_sentiment_metrics_fn(list_hyp, list_label)
        print("(Epoch {}) VALID LOSS:{:.4f} {}".format((epoch+1),
            total_loss/(i+1), metrics_to_string(metrics)))
    return base_model

def eval_model(model, test_loader, i2w):
    # Evaluate on test
    model.eval()
    torch.set_grad_enabled(False)

    total_loss, total_correct, total_labels = 0, 0, 0
    list_hyp, list_label = [], []

    pbar = tqdm(test_loader, leave=True, total=len(test_loader))
    for i, batch_data in enumerate(pbar):
        _, batch_hyp, _ = forward_sequence_classification(model, batch_data[:-1], i2w=i2w, device='cuda')
        list_hyp += batch_hyp

    # Save prediction
    df = pd.DataFrame({'label':list_hyp}).reset_index()
    df.to_csv('pred.txt', index=False)

    print(df)

    



def attack(text_ls, true_label, predictor, tokenizer, sim_score_threshold=0.5, sim_score_window=15, batch_size=32, 
           import_score_threshold=-1.):
    label_dict = {
        'positive': 0, 
        'neutral': 1, 
        'negative': 2}
    
    subwords = tokenizer.encode(text_ls)
    subwords = torch.LongTensor(subwords).view(1, -1).to(predictor.device)

    logits = predictor(subwords)[0]
    orig_label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
    
    orig_probs = F.softmax(logits, dim=-1).squeeze()
    orig_prob = F.softmax(logits, dim=-1).squeeze()[orig_label].detach().cpu().numpy()
    
    ic(orig_label)
    ic(orig_prob)
    
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        text_ls = word_tokenize(text_ls)
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1 # kecilkan similarity threshold
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1
        
        leave_1_texts = [' '.join(text_ls[:ii] + [tokenizer.mask_token] + text_ls[min(ii + 1, len_text):]) for ii in range(len_text)]
        
#         ic(leave_1_texts)
        
        leave_1_probs = []
        leave_1_probs_argmax = []
        num_queries += len(leave_1_texts)
        for text_leave_1 in leave_1_texts:
            subwords_leave_1 = tokenizer.encode(text_leave_1)
            subwords_leave_1 = torch.LongTensor(subwords_leave_1).view(1, -1).to(predictor.device)
            logits_leave_1 = predictor(subwords_leave_1)[0]
            orig_label_leave_1 = torch.topk(logits_leave_1, k=1, dim=-1)[1].squeeze().item()
            
            leave_1_probs_argmax.append(orig_label_leave_1)
            leave_1_probs.append(F.softmax(logits_leave_1, dim=-1).squeeze().detach().cpu().numpy())
            
        leave_1_probs = torch.tensor(leave_1_probs).to("cuda:0")
        
#         ic(leave_1_probs)
        orig_prob_extended=np.empty(len_text)
        orig_prob_extended.fill(orig_prob)
        orig_prob_extended = torch.tensor(orig_prob_extended).to("cuda:0")
        
        arr1 = orig_prob_extended - leave_1_probs[:,orig_label] + float(leave_1_probs_argmax != orig_label)
        
#         ic(orig_probs)
#         ic(orig_prob_extended)
#         ic(leave_1_probs_argmax)
#         ic(orig_probs[leave_1_probs_argmax])
#         leave_1_probs_argmax = torch.tensor(leave_1_probs_argmax).to("cuda:0")
#         ic(torch.index_select(orig_probs, 0, leave_1_probs_argmax))
        
        arr2 = (leave_1_probs.max(dim=-1)[0].to("cuda:0") - orig_probs[leave_1_probs_argmax])
        
#         ic(arr1)
#         ic(arr2)

        import_scores = arr1*arr2
    
#         ic(sorted(enumerate(import_scores), key=lambda x: x[1], reverse=False))
#         ic(sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True))
        
        ic(import_scores)
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=False):
            try:
                if score > import_score_threshold and text_ls[idx] not in stop_words_set:
                    ic(text_ls[idx],score)
                    words_perturb.append((idx, text_ls[idx]))
            except Exception as e:
                print(e)
                print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))
        ic(words_perturb)