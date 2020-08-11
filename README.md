# rcv_badge_fastser_rcnn

![image](https://user-images.githubusercontent.com/62923434/89851802-54bf8080-dbc8-11ea-852b-8df692035c82.png)
![image](https://user-images.githubusercontent.com/62923434/89851827-63a63300-dbc8-11ea-92ed-8072c616da07.png)

## more details
- Dataset\
train=Pascal VOC 2007 trainval(~5011)\
test=Pascal VOC 2007 test
- Data augumentation\
1.randomly horizontal flip\
2.optional resize(min size=600,max size=1000)\
3.Do not use difficult flaged image
- train\
1.batch size=1\
2.epoch=~20
