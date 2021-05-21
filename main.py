from platform import python_version
import torch
from datasets import load_dataset

from preprocess import *
from preprocessed_data import *
from misc import *
from make_batch import *
from validation import *
from train import *
from basic_lstm import *
from lstm_bidirec import *
from lstm_with_attention import *
from lstm_attn_lstm import *



def main():
	print("python", python_version())
	print("torch", torch.__version__)

	squad_dataset = load_dataset('squad')
	print(squad_dataset['train'][0])

	train_data, del_train_data_index, valid_data, del_valid_data_index = preprocessing(squad_data)

	
	