from dataSource import Dataset,LEM,CampoVerde,DataSource,SARSource
import numpy as np
import matplotlib.pyplot as plt

class DatasetStats():
    def __init__(self,dataset):
        self.dataset=dataset
    
    def calcAverageTimeseries(self,ims,mask):
        time_delta=self.dataset.getTimeDelta()
        print(time_delta)
        for channel in range(self.dataset.getBandN()):
            averageTimeseries=[]
            for t_step in range(0,self.dataset.t_len):
                im=ims[t_step,:,:,channel]
                #mask_t=mask[t_step]
                
                #print("im shape: {}, mask shape: {}".format(im.shape,mask.shape))
                im=im.flatten()
                mask_t=mask.flatten()
                #print("im shape: {}, mask shape: {}".format(im.shape,mask.shape))

                im=im[mask_t>0] # only train and test pixels (1 and 2)
                averageTimeseries.append(np.average(im))
            averageTimeseries=np.asarray(averageTimeseries)
            plt.figure(channel)
            fig, ax = plt.subplots()
            ax.plot(time_delta,averageTimeseries,marker=".")
            ax.set(xlabel='time ID', ylabel='band',title='Image average over time')
            plt.grid()
            print('averageTimeseries',averageTimeseries)
        plt.show()
            



