# make dataloader 


def preprocessing(squad_dataset):
	train_data = []
	del_train_data_index = []
	count = 0
	for i, data in enumerate(squad_dataset['train']):
	    try:
	        train_data.append(advanced_preprocess1(data))
	    except:
	        train_data.append([])
	        del_train_data_index.append(i)
	        count+=1
	        

	valid_data = []
	del_valid_data_index = []
	count = 0
	for i, data in enumerate(squad_dataset['validation']):
	    try:
	        valid_data.append(advanced_preprocess1(data))
	    except:
	        valid_data.append([])
	        del_valid_data_index.append(i)
	        count += 1

	return train_data, del_train_data_index, valid_data, del_valid_data_index
	