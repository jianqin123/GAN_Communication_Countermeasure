# GAN_Communication_Countermeasure
>signal reconstruction is significant to communication countermeasure,there we focuse on the algorithm of signal reconstrction.<br/>
>>1,a neural network is build to claify the modulation of signals,which can reach the accuracy of 98% when signal snr is betweeen -15db and 15db.<br/>
2,using GAN to regeneration signal similar to the original signal ,like some application in Image Generation,GAN,WGAN,LSGAN were studied. <br/>
3,Conditional GAN is applied to generation signal with some different modulation signals mixed,this algorithm can produce signal on the basis of features. <br/>

## the architecture of the algorithem <br>
> a GAN architecture is build like below,the loss functions includs GAN,WGAN,LSGAN <br/>
 ![image](https://github.com/jianqin123/GAN_Communication_Countermeasure/blob/master/imag/architecture%20of%20GAN%20to%20generate%20signal.png)<br/>
> both the generator and discriminator is based on DNN,whose architectures are shown below<br/>
![image](https://github.com/jianqin123/GAN_Communication_Countermeasure/blob/master/imag/neural%20network%20of%20discriminator%20and%20generator.png)
### reconstrction of AM signal
>>>>> ![image](https://github.com/jianqin123/GAN_Communication_Countermeasure/blob/master/imag/AM_regenaration.png)<br/>
## architecture of conditional GAN<br/>
>>>>> ![image](https://github.com/jianqin123/GAN_Communication_Countermeasure/blob/master/imag/conditiaon_GAN.png)<br/>




## file directory description 
> data: generated signal data of different modulation  <br>
> imag:picture used in readme <br>
> log :log info of different modulations signal regeneration algorithm <br>
> model ：model meta info which can be recovery when needed <br>
> result :generated signal data shown by picture<br>
> src : source code  <br>
>> Analog_Clarify.py: clarify signal modulation by traditional methods <br>
>> Analog_signal_LSGAN.py: regeneration signal data by LSGAN<br>
>> Signal_Generation_Param_select_2.py: select best params of GAN<br>
>> Signal_Regeneration_GAN: regeneration signal data by GAN ，which can produce different signal data through different model <br/>
>> modulation.py :clarify signal modulation by neural network <br>




