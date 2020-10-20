from dataSource import Dataset,LEM,CampoVerde,DataSource,SARSource
import numpy as np
import matplotlib.pyplot as plt

class DatasetStats():
    def __init__(self,dataset):
        self.dataset=dataset
    
    def calcAverageTimeseries(self,ims,mask):
        averageTimeseries=[]
        channel=0
        for t_step in range(0,self.dataset.t_len):
            im=ims[t_step,:,:,channel]
            #mask_t=mask[t_step]
            
            print("im shape: {}, mask shape: {}".format(im.shape,mask.shape))
            im=im.flatten()
            mask_t=mask.flatten()
            print("im shape: {}, mask shape: {}".format(im.shape,mask.shape))

            im=im[mask_t>0] # only train and test pixels (1 and 2)
            averageTimeseries.append(np.average(im))
        averageTimeseries=np.asarray(averageTimeseries)
        fig, ax = plt.subplots()
        ax.plot(averageTimeseries)
        ax.set(xlabel='time ID', ylabel='band',title='Image average over time')
        ax.grid()
        plt.show()
        print('averageTimeseries',averageTimeseries)




