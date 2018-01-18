# GAN_Communication_Countermeasure
>signal reconstruction is significant to communication countermeasure,there we focuse on the algorithm of signal reconstrction.<br/>
>>1,a neural network is build to claify the modulation of signals,which can reach the accuracy of 98% when signal snr is betweeen -15db and 15db.<br/>
2,using GAN to regeneration signal similar to the original signal ,like some application in Image Generation,GAN,WGAN,LSGAN were studied. <br/>
3,Conditional GAN is applied to generation signal with some different modulation signals mixed,this algorithm can produce signal on the basis of features. <br/>

## the architecture of the algorithem <br>
> a GAN architecture is build like below,the loss functions includs GAN,WGAN,LSGAN <br/>
  >>>>>![image](https://github.com/jianqin123/GAN_Communication_Countermeasure/blob/master/imag/Signal_Generation_GAN_architecture.png)<br/>
> both the generator and discriminator is based on DNN,whose architectures are shown below<br/>
### reconstrction of AM signal
>>>>> ![image](https://github.com/jianqin123/GAN_Communication_Countermeasure/blob/master/imag/AM_regenaration.png)<br/>
## architecture of conditional GAN<br/>
>>>>> ![image](https://github.com/jianqin123/GAN_Communication_Countermeasure/blob/master/imag/conditiaon_GAN.png)<br/>


