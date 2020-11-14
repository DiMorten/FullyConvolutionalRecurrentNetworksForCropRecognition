 
import numpy as np
import cv2
import glob
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-ds', '--dataset', dest='dataset',
					default='cv', help='t len')
parser.add_argument('-mdl', '--model', dest='model',
					default='unet', help='t len')

a = parser.parse_args()

dataset=a.dataset
model=a.model


#locations_folder='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/batched_rnn_src/'
#label_checker=np.load('cv/locations_label.npy')

#folder='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/hn_data/results/convlstm_16_300_nompool/'
folder='/home/lvc/Jorg/deep_learning/LSTM-Final-Project/cv_data/buffer/seq1/15/convlstm/'

def patch_file_id_order_from_folder(folder_path):
	paths=glob.glob(folder_path+'*.npy')
	
	order=[int(paths[i].partition('patch_')[2].partition('_')[0]) for i in range(len(paths))]
	print(len(order))
	print(order[0:20])
	return order
	#deb.prints(len(paths))
	#for path in paths:
		#print(path)
	#	files.append(np.load(path))
	#return np.asarray(files),paths

# == normy3

#path='/home/lvc/Jorg/igarss/convrnn_remote_sensing/results/seq2seq_ignorelabel/'
#path='../model/'
path='../'

#dataset='lm'
#model='biconvlstm'
#model='convlstm'
#model='densenet'
data_path='../../../../../dataset/dataset/'
if dataset=='lm':
	path+='lm/'
	if model=='densenet':
		predictions_path=path+'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'
	elif model=='biconvlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy'
	elif model=='convlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_batch16_full.npy'
	elif model=='unet':
		predictions_path=path+'prediction_BUnet4ConvLSTM_repeating1.npy'
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating2.npy'
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating4.npy'


	elif model=='atrous':
		predictions_path=path+'prediction_BAtrousConvLSTM_2convins5.npy'
	elif model=='atrousgap':
		predictions_path=path+'prediction_BAtrousGAPConvLSTM_raulapproved.npy'
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating3.npy'
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating4.npy'
		


	mask_path=data_path+'lm_data/TrainTestMask.tif'
	location_path=data_path+'src_seq2seq/locations/lm/'
	folder_load_path=data_path+'lm_data/train_test/test/labels/'

	custom_colormap = np.array([[255,146,36],
					[255,255,0],
					[164,164,164],
					[255,62,62],
					[0,0,0],
					[172,89,255],
					[0,166,83],
					[40,255,40],
					[187,122,83],
					[217,64,238],
					[0,113,225],
					[128,0,0],
					[114,114,56],
					[53,255,255]])
elif dataset=='cv':

	path+='cv/'
	if model=='densenet':
		predictions_path=path+'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'
	elif model=='biconvlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy'
	elif model=='convlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_batch16_full.npy'		
	elif model=='unet':
#		predictions_path=path+'prediction_BUnet4ConvLSTM_repeating2.npy' this is paper version
		predictions_path=path+'prediction_BUnet4ConvLSTM_repeating4.npy'				
	elif model=='atrous':
		predictions_path=path+'prediction_BAtrousConvLSTM_repeating2.npy'			
	elif model=='atrousgap':
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_raulapproved.npy'			
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating4.npy'			
		predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating6.npy'			
	elif model=='unetend':
		predictions_path=path+'prediction_unet_convlstm_temouri2.npy'			
	elif model=='allinputs':
		predictions_path=path+'prediction_bconvlstm_wholeinput.npy'			

	mask_path=data_path+'cv_data/TrainTestMask.tif'
	location_path=data_path+'cv_data/locations/'

	folder_load_path=data_path+'cv_data/train_test/test/labels/'

	custom_colormap = np.array([[255, 146, 36],
                   [255, 255, 0],
                   [164, 164, 164],
                   [255, 62, 62],
                   [0, 0, 0],
                   [172, 89, 255],
                   [0, 166, 83],
                   [40, 255, 40],
                   [187, 122, 83],
                   [217, 64, 238],
                   [45, 150, 255]])

order_id=patch_file_id_order_from_folder(folder_load_path)

cols=np.load(location_path+'locations_col.npy')
rows=np.load(location_path+'locations_row.npy')

cols=cols[order_id]
rows=rows[order_id]
labels=np.load(path+'labels.npy').argmax(axis=4)
predictions=np.load(predictions_path).argmax(axis=4)

class_n=np.max(predictions)+1
print("class_n",class_n)
labels[labels==class_n]=255 # background

# Print stuff
print(cols.shape)
print(rows.shape)
print(labels.shape)
print(predictions.shape)
print("np.unique(labels,return_counts=True)",
	np.unique(labels,return_counts=True))
print("np.unique(predictions,return_counts=True)",
	np.unique(predictions,return_counts=True))

# Specify variables
sequence_len=labels.shape[1]
patch_len=labels.shape[2]

# Load mask
mask=cv2.imread(mask_path,-1)
mask[mask==1]=0 # training as background
print("Mask shape",mask.shape)
#print((sequence_len,)+mask.shape)

# Reconstruct the image

label_rebuilt=np.ones(((sequence_len,)+mask.shape)).astype(np.uint8)*255
prediction_rebuilt=np.ones(((sequence_len,)+mask.shape)).astype(np.uint8)*255
print("label_rebuilt.shape",label_rebuilt.shape)
for row,col,label,prediction in zip(rows,cols,labels,predictions):
	label_rebuilt[:,row:row+patch_len,col:col+patch_len]=label.copy()
	prediction_rebuilt[:,row:row+patch_len,col:col+patch_len]=prediction.copy()

print(np.unique(label_rebuilt,return_counts=True))
print(np.unique(prediction_rebuilt,return_counts=True))


# everything outside mask is 255
for t_step in range(sequence_len):
	label_rebuilt[t_step][mask==0]=255

	prediction_rebuilt[t_step][mask==0]=255
#label_rebuilt[label_rebuilt==class_n]=255
	
print("everything outside mask is 255")
print(np.unique(label_rebuilt,return_counts=True))
print(np.unique(prediction_rebuilt,return_counts=True))


# Paint it!


print(custom_colormap.shape)
#class_n=custom_colormap.shape[0]
#=== change to rgb
print("Gray",prediction_rebuilt.dtype)
prediction_rgb=np.zeros((prediction_rebuilt.shape+(3,))).astype(np.uint8)
label_rgb=np.zeros_like(prediction_rgb)

for t_step in range(sequence_len):
	prediction_rgb[t_step]=cv2.cvtColor(prediction_rebuilt[t_step],cv2.COLOR_GRAY2RGB)
	label_rgb[t_step]=cv2.cvtColor(label_rebuilt[t_step],cv2.COLOR_GRAY2RGB)

print("RGB",prediction_rgb.dtype,prediction_rgb.shape)

for idx in range(custom_colormap.shape[0]):
	print("Assigning color. t_step:",idx)
	for chan in [0,1,2]:
		prediction_rgb[:,:,:,chan][prediction_rgb[:,:,:,chan]==idx]=custom_colormap[idx,chan]
		label_rgb[:,:,:,chan][label_rgb[:,:,:,chan]==idx]=custom_colormap[idx,chan]

print("RGB",prediction_rgb.dtype,prediction_rgb.shape)

#for idx in range(custom_colormap.shape[0]):
#	for chan in [0,1,2]:
#		prediction_rgb[:,:,chan][prediction_rgb[:,:,chan]==correspondence[idx]]=custom_colormap[idx,chan]

for t_step in range(sequence_len):

	label_rgb[t_step]=cv2.cvtColor(label_rgb[t_step],cv2.COLOR_BGR2RGB)
	prediction_rgb[t_step]=cv2.cvtColor(prediction_rgb[t_step],cv2.COLOR_BGR2RGB)

	cv2.imwrite(dataset+"/"+model+"/prediction_t"+str(t_step)+"_"+model+".png",prediction_rgb[t_step])
	cv2.imwrite(dataset+"/"+model+"/label_t"+str(t_step)+"_"+model+".png",label_rgb[t_step])

print(prediction_rgb[0,0,0,:])
if False:

	print(cols.shape)



	# Transform back

	"""
	correspondence=np.array([0,1],
							[1,2],
							[2,3],
							[3,4],
							[4,6],
							[5,7],
							[6,8],
							[7,9],
							[8,10],
							[9,11])
	"""
	correspondence=np.array([ 1 , 2 , 3 , 4 , 6 , 7,  8 , 9 ,10 ,11])

	print(np.all(labels==label_checker))

	print(label_checker.shape,labels.dtype)
	print(np.unique(label_checker,return_counts=True))


	print("Labels",labels.shape,labels.dtype)
	print(np.unique(labels,return_counts=True))

	print(predictions.shape,predictions.dtype)
	print(np.unique(predictions,return_counts=True))

	labels_tmp=labels.copy()
	predictions_tmp=predictions.copy()


	for idx in range(correspondence.shape[0]):
		labels_tmp[labels==idx]=correspondence[idx]
		predictions_tmp[predictions==idx]=correspondence[idx]

	labels=labels_tmp.copy()
	predictions=predictions_tmp.copy()
	#print(np.all(labels==label_checker))


	print("Labels",labels.shape,labels.dtype)
	print(np.unique(labels,return_counts=True))

	print(predictions.shape,predictions.dtype)
	print(np.unique(predictions,return_counts=True))

	#labels=labels+1
	#predictions=predictions+1



	mask=cv2.imread(mask_path,-1)
	print("Mask shape",mask.shape)

	label_rebuilt=np.ones_like(mask).astype(np.uint8)*255
	prediction_rebuilt=np.ones_like(mask).astype(np.uint8)*255
	for row,col,label,prediction in zip(rows,cols,labels,predictions):
		label_rebuilt[row,col]=label
		prediction_rebuilt[row,col]=prediction

	print(np.unique(label_rebuilt,return_counts=True))
	print(np.unique(prediction_rebuilt,return_counts=True))


	# [ 1  2  3  4  6  7  8  9 10 11]


	custom_colormap = np.array([[255, 146, 36],
	                   [255, 255, 0],
	                   [164, 164, 164],
	                   [255, 62, 62],
	                   [0, 0, 0],
	                   [172, 89, 255],
	                   [0, 166, 83],
	                   [40, 255, 40],
	                   [187, 122, 83],
	                   [217, 64, 238],
	                   [45, 150, 255]])
	custom_colormap_tmp=custom_colormap.copy()

	custom_colormap=custom_colormap[np.array([0,1,2,3,5,6,7,8,9,10])]
	print(custom_colormap.shape)

	#=== change to rgb
	print("Gray",prediction_rebuilt.dtype)

	prediction_rgb=cv2.cvtColor(prediction_rebuilt,cv2.COLOR_GRAY2RGB)
	print("RGB",prediction_rgb.dtype)

	print(prediction_rgb.shape)


	for idx in range(custom_colormap.shape[0]):
		for chan in [0,1,2]:
			prediction_rgb[:,:,chan][prediction_rgb[:,:,chan]==correspondence[idx]]=custom_colormap[idx,chan]

	prediction_rgb=cv2.cvtColor(prediction_rgb,cv2.COLOR_BGR2RGB)

	# == label]

	label_rgb=cv2.cvtColor(label_rebuilt,cv2.COLOR_GRAY2RGB)
	print("RGB",label_rgb.dtype)

	print(label_rgb.shape)


	for idx in range(custom_colormap.shape[0]):
		for chan in [0,1,2]:
			label_rgb[:,:,chan][label_rgb[:,:,chan]==correspondence[idx]]=custom_colormap[idx,chan]

	label_rgb=cv2.cvtColor(label_rgb,cv2.COLOR_BGR2RGB)


	cv2.imwrite("reconstruction_normy3.png",prediction_rgb)
	cv2.imwrite("label_normy3.png",label_rgb)

	print(prediction_rgb[0,0,:])