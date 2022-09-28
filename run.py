"""BERT finetuning runner."""
from __future__ import absolute_import, division, print_function
import argparse
import os
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE,cached_path
from _model import _model, BertConfig, WEIGHTS_NAME, CONFIG_NAME
#from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from _utils import *
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import metrics
from torch import nn
from temping import get_temps
from transformers import (
                AdamW,
                get_scheduler,
                set_seed,
                get_linear_schedule_with_warmup,
                BertTokenizer
                )
#单GPU
os.environ["CUDA_VISIBLE_DEVICES"]="3"


"""BERT finetuning runner."""
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default='MOSI',choices=["MOSI", "MOSEI"], type=str)

    ## Required parameters
    parser.add_argument("--data_dir", default='/home/MOSI/text', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='BERT', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name", default='multi', type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir", default='output', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=50, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",default=True,
                        help="Whether to run training.'store_true'")
    parser.add_argument("--do_test", default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--alpha", default=0.3, type=float,
                    help="weight of loss and loss_c")

    parser.add_argument("--q_size", default=100000, type=int,
                help="size of buffer queue")
    parser.add_argument("--contrastive", action='store_true',
                help="whether contrastive or not")

    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--test_batch_size", default=32, type=int,
                        help="Total batch size for test.")
    parser.add_argument("--learning_rate", default=5e-6, type=float,
                        help="The initial learning rate for Adam.5e-5")
    parser.add_argument("--num_train_epochs", default=12, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=11111,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--dropout", default=0.3, type=float,
                        help="Total batch size for training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                    help="Total batch size for training.")
    args = parser.parse_args()
    processors = {
        "multi": PgProcessor,
    }

    num_labels_task = {
        "multi": 1
    }

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = 1
    #logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    #seed_num = np.random.randint(1,10000)
    seed_num = args.seed
    random.seed(args.seed)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed_num)

    if not args.do_train and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_test` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train==True:
        train_examples = processor.get_train_examples(args.dataset+"/text")
        num_train_optimization_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format("-1"))
    ##############################################################################################################
    model = _model.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels = num_labels, q_size=args.q_size, alpha=args.alpha, contrastive=args.contrastive,dropout=args.dropout)
    # Freezing all layer except for last transformer layer and its follows  

    # for name, param in model.named_parameters():  
    #     # print(name,end=" ")
    #     # print(param.requires_grad)
    #     if "bert" in name:
    #         param.requires_grad = False
    #     param.requires_grad = False
    #     if "encoder.layer.0" in name or "encoder.layer.1" in name:
    #         param.requires_grad = True
    #     if "encoder.layer.2" in name or "encoder.layer.3" in name :
    #         param.requires_grad = True
    #     if "encoder.layer.4" in name or  "encoder.layer.5" in name:
    #         param.requires_grad = True
    #     if "encoder.layer.6" in name or "encoder.layer.7" in name:
    #         param.requires_grad = True
    #     if "encoder.layer.8" in name or "encoder.layer.9" in name :
    #         param.requires_grad = True
    #     if "encoder.layer.10" in name or  "encoder.layer.11" in name:
    #         param.requires_grad = True
    #     if "BertFinetun" in name or "pooler" in name:
    #         param.requires_grad = True
    ##############################################################################################################
    model.to(device)

    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    new_decay = ['BertFine']

    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not any(np in n for np in new_decay)], 'weight_decay': 0.2},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay )and any(np in n for np in new_decay)],'lr':args.learning_rate}
    # ]
    optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if "bert" in n], 'weight_decay': 0.15},
    {'params': [p for n, p in param_optimizer if "bert" not in n], 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    train_audio,valid_audio,test_audio= pickle.load(open(args.dataset+'/audio/paudio.pickle','rb'))

    train_audio_IS,valid_audio_IS,test_audio_IS= pickle.load(open(args.dataset+'/audio/paudio_IS.pickle','rb'))

    train_video,valid_video,test_video= pickle.load(open(args.dataset+'/video/pvideo.pickle','rb'))

    #train_speaker,valid_speaker,test_speaker= pickle.load(open('/home/MOSI/pspeaker.pickle','rb'))

    valid_audio = test_audio
    valid_audio_IS = test_audio_IS
    valid_video = test_video
    #valid_speaker = test_speaker
    corr_list=[]
    mae_list=[]
    if args.do_train==True:
        #print(250)
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
        
        #logger.info("  Num examples = %d", len(train_examples))
        #logger.info("  Batch size = %d", args.train_batch_size)
        #logger.info("  Num steps = %d", num_train_optimization_steps)
        all_train_audio = torch.tensor(train_audio, dtype=torch.float32)
        all_train_video = torch.tensor(train_video, dtype=torch.float32)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float32)

        if args.dataset=="MOSI":
            all_train_audio_IS = torch.tensor([f[1] for f in train_audio_IS], dtype=torch.float32)
        elif args.dataset=="MOSEI":
            all_train_audio_IS = torch.tensor([f[0] for f in train_audio_IS], dtype=torch.float32)
        #all_train_speaker = torch.tensor(train_speaker, dtype=torch.float32)

        # all_train_audio_IS_1 = torch.tensor([f[0] for f in train_audio_IS], dtype=torch.float32)
        # all_train_audio_IS = torch.cat((all_train_audio_IS, all_train_audio_IS_1), dim=1)

        all_train_audio_IS = all_train_audio_IS.unsqueeze(1)



        # print(all_input_ids.shape)
        # print(all_input_mask.shape)
        # print(all_segment_ids.shape)
        # print(all_train_audio.shape)
        # print(all_label_ids.shape)
        # print(all_train_audio_IS.shape)
        # print(all_train_video.shape)
        #print(all_train_speaker.shape)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_train_audio, all_label_ids, all_train_audio_IS, all_train_video)
        train_sampler = RandomSampler(train_data)
        #train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        ## Evaluate for each epcoh
        eval_examples = processor.get_dev_examples(args.dataset+"/text")
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
        all_valid_audio = torch.tensor(valid_audio, dtype=torch.float32,requires_grad=True)
        all_valid_video = torch.tensor(valid_video, dtype=torch.float32,requires_grad=True)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float32)

        if args.dataset=="MOSI":
            all_valid_audio_IS = torch.tensor([f[1] for f in valid_audio_IS], dtype=torch.float32)
        elif args.dataset=="MOSEI":
            all_valid_audio_IS = torch.tensor([f[0] for f in valid_audio_IS], dtype=torch.float32)
        #all_valid_speaker = torch.tensor(valid_speaker, dtype=torch.float32,requires_grad=True)
        # all_valid_audio_IS_1 = torch.tensor([f[0] for f in valid_audio_IS], dtype=torch.float32)
        # all_valid_audio_IS = torch.cat((all_valid_audio_IS, all_valid_audio_IS_1), dim=1)
        all_valid_audio_IS = all_valid_audio_IS.unsqueeze(1)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,all_valid_audio,all_label_ids, all_valid_audio_IS, all_valid_video)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)
        max_acc = 0
        min_loss = 100
        # 默认方法

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("***** Running training *****")
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, train_audio1, label_ids, train_audio_IS1, train_video1 = batch
                loss,_ = model(input_ids, train_audio1,segment_ids, input_mask, label_ids, train_audio_IS1, train_video1)
                # if n_gpu > 1:
                #     loss = loss.mean() # mean() to average on multi-gpu.
                #     print(260)
                # if args.gradient_accumulation_steps > 1:
                #     print(260)
                #     loss = loss / args.gradient_accumulation_steps
                #loss = (loss-0.3).abs()+0.3
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # jjj=1
                # for name, parms in model.named_parameters():
                #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad)
                #     jjj+=1
                #     if jjj==10:
                #         break

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    global_step += 1
            logger.info("***** Running evaluation *****")
            #logger.info("  Num examples = %d", len(eval_examples))
            #logger.info("  Batch size = %d", args.eval_batch_size)
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            predict_list = []
            truth_list = []
            for input_ids, input_mask, segment_ids,valid_audio1,label_ids, valid_audio_IS1, valid_video1 in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                valid_audio1 = valid_audio1.to(device)
                valid_audio_IS1 = valid_audio_IS1.to(device)
                valid_video1 = valid_video1.to(device)
                #valid_speaker1 = valid_speaker1.to(device)
                with torch.no_grad():
                    tmp_eval_loss,logits = model(input_ids, valid_audio1,segment_ids, input_mask,label_ids,valid_audio_IS1, valid_video1)
                    #logits,_,_ = model(input_ids,valid_audio1, segment_ids, input_mask,IS=valid_audio_IS1, video=valid_video1)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                # print(logits)
                # print(250)

                tmp_eval_accuracy = accuracy1(logits, label_ids)
                for i in range(len(logits)):
                    predict_list.append(logits[i])
                    truth_list.append(label_ids[i])
                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1
            predict_list = np.array(predict_list).reshape(-1)
            truth_list = np.array(truth_list)
            corr = np.corrcoef(predict_list, truth_list)[0][1]
            mae = np.mean(np.absolute(predict_list - truth_list))
            corr_list.append(corr)
            mae_list.append(mae)
            eval_loss = eval_loss / nb_eval_steps
            #eval_accuracy = eval_accuracy / nb_eval_examples
            if args.dataset=="MOSI":
                eval_accuracy = eval_accuracy / 656
            elif args.dataset=="MOSEI":
                eval_accuracy = eval_accuracy / 3615
            loss = tr_loss/nb_tr_steps if args.do_train else None
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'global_step': global_step,
                      'loss': loss}
            
            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

            # Save a trained model and the associated configuration
            #print(eval_loss)
            #if eval_loss<min_loss:
            #if loss < min_loss:
            if eval_accuracy>max_acc:
                #min_loss = eval_loss
                #min_loss = loss
                max_acc=eval_accuracy
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                torch.save(model_to_save.state_dict(), output_model_file)
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                with open(output_config_file, 'w') as f:
                    f.write(model_to_save.config.to_json_string())

    if args.do_test==True:
        ## Evaluate for each epcoh
        test_examples = processor.get_test_examples(args.dataset+"/text")
        test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("")
        logger.info("***** Running test *****")
        #logger.info("  Num examples = %d", len(test_examples))
        #logger.info("  Batch size = %d", args.test_batch_size)
        all_test_audio = torch.tensor(test_audio, dtype=torch.float32,requires_grad=True)
        all_test_video = torch.tensor(test_video, dtype=torch.float32,requires_grad=True)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.float32)

        if args.dataset=="MOSI":
            all_test_audio_IS = torch.tensor([f[1] for f in test_audio_IS], dtype=torch.float32)
        elif args.dataset=="MOSEI":
            all_test_audio_IS = torch.tensor([f[0] for f in test_audio_IS], dtype=torch.float32)
        #all_test_speaker = torch.tensor(test_speaker, dtype=torch.float32,requires_grad=True)
        # all_test_audio_IS_1 = torch.tensor([f[0] for f in test_audio_IS], dtype=torch.float32)
        # all_test_audio_IS = torch.cat((all_test_audio_IS, all_test_audio_IS_1), dim=1)
        all_test_audio_IS = all_test_audio_IS.unsqueeze(1)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_test_audio, all_test_audio_IS, all_test_video)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.train_batch_size)
        model = _model.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels = num_labels, q_size=args.q_size, alpha=args.alpha,  contrastive=args.contrastive,dropout=args.dropout)
        model.load_state_dict(torch.load('output/pytorch_model.bin'))
        model.to(device)
        model.eval()
        test_loss, test_accuracy = 0, 0
        nb_test_steps, nb_test_examples = 0, 0
        predict_list = []
        truth_list = []
        text_attention_list = []
        fusion_attention_list = []
        with torch.no_grad():
            for input_ids, input_mask, segment_ids, label_ids, test_audio1, test_audio_IS1, test_video1 in tqdm(test_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                test_audio1 = test_audio1.to(device)
                test_audio_IS1 = test_audio_IS1.to(device)
                test_video1  = test_video1.to(device)
                #test_speaker1  = test_speaker1.to(device)
                with torch.no_grad():
                    tmp_test_loss,logits = model(input_ids, test_audio1,segment_ids, input_mask, label_ids, test_audio_IS1, test_video1)
                    #logits,text_attention,fusion_attention = model(input_ids, test_audio1,segment_ids, input_mask, IS=test_audio_IS1, video=test_video1)
                
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                #text_attention = text_attention.cpu().numpy()
                #fusion_attention = fusion_attention.cpu().numpy()
                test_loss += tmp_test_loss.mean().item()

                for i in range(len(logits)):
                    predict_list.append(logits[i])
                    truth_list.append(label_ids[i])
                    #text_attention_list.append(text_attention[i])
                    #fusion_attention_list.append(fusion_attention[i])
                nb_test_examples += input_ids.size(0)
                nb_test_steps += 1
        
        exclude_zero = False
        non_zeros = np.array([i for i, e in enumerate(truth_list) if e != 0 ])
        predict_list = np.array(predict_list).reshape(-1)
        truth_list = np.array(truth_list)
        predict_list1 = (predict_list[non_zeros] > 0)
        truth_list1 = (truth_list[non_zeros] > 0)
        test_loss = test_loss / nb_test_steps
        test_preds_a7 = np.clip(predict_list, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(truth_list, a_min=-3., a_max=3.)
        acc7 = accuracy_7(test_preds_a7,test_truth_a7)
        f_score = f1_score(predict_list1, truth_list1, average='weighted')

        acc = accuracy_score(truth_list1, predict_list1)
        corr=my_max(corr_list)
        mae=my_min(mae_list)
        loss = tr_loss/nb_tr_steps if args.do_train==True else None
        results = {
                  'acc':acc,
                  'F1':f_score,
                  'mae':mae,
                  'corr':corr}
        logger.info("***** test results *****")
        print(results)

        return results, args.train_batch_size, args.learning_rate, args.alpha
        

if __name__ == "__main__":

    os.system('mkdir output')
    results, batch_size, lr, alpha = main()
    #os.system('rm -r output')


