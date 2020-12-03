import chainer
import chainer.functions as F
import chainer.links as L
import pdb

class PFHDRNet(chainer.Chain):

    def __init__(self, train=True, in_channels=3):
        w = chainer.initializers.Normal(0.02)
        super(PFHDRNet, self).__init__(
            c0 = L.Convolution2D(in_channels, 64, 4, stride=2, pad=1,initialW=w),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1,initialW=w),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1,initialW=w),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1,initialW=w),
            c4 = L.Convolution2D(512, 512, 4, stride=2, pad=1,initialW=w),
            c5 = L.Convolution2D(512, 512, 4, stride=2, pad=1,initialW=w),
            c6 = L.Convolution2D(512, 512, 4, stride=2, pad=1,initialW=w),
            c7 = L.Convolution2D(512, 512, 4, stride=2, pad=1,initialW=w),
            c8 = L.Convolution2D(512, 512, 4, stride=2, pad=1,initialW=w),


            dc00 = L.DeconvolutionND(3, 512, 512, (4, 4, 4), stride=(2,2,2), pad=1,initialW=w),
            dc0 = L.DeconvolutionND(3, 1024, 512, (4, 4, 4), stride=(2,2,2), pad=1,initialW=w),
            dc1 = L.ConvolutionND(3, 1024, 512, (4, 4, 4), stride=(2,2,2), pad=1,initialW=w),
            dc2 = L.ConvolutionND(3, 1024, 512, (3, 4, 4), stride=(1,2,2), pad=1,initialW=w),
            dc3 = L.ConvolutionND(3, 1024, 512, (3, 4, 4), stride=(1,2,2), pad=1,initialW=w),
            dc4 = L.ConvolutionND(3, 1024, 256, (3, 4, 4), stride=(1,2,2), pad=1,initialW=w),
            dc5 = L.ConvolutionND(3, 512, 128, (3, 4, 4), stride=(1,2,2), pad=1,initialW=w),
            dc6 = L.DeconvolutionND(3, 256, 64, (3, 4, 4), stride=(1,2,2), pad=1,initialW=w),
            dc7 = L.DeconvolutionND(3, 128, in_channels, (3, 4, 4), stride=(1,2,2), pad=1,initialW=w),

                                    
            bnc1 = L.BatchNormalization(128),
            bnc2 = L.BatchNormalization(256),
            bnc3 = L.BatchNormalization(512),
            bnc4 = L.BatchNormalization(512),
            bnc5 = L.BatchNormalization(512),
            bnc6 = L.BatchNormalization(512),
            bnc7 = L.BatchNormalization(512),
            bnc8 = L.BatchNormalization(512),

            bndc00 = L.BatchNormalization(512),
            bndc0 = L.BatchNormalization(512),
            bndc1 = L.BatchNormalization(512),
            bndc2 = L.BatchNormalization(512),
            bndc3 = L.BatchNormalization(512),
            bndc4 = L.BatchNormalization(256),
            bndc5 = L.BatchNormalization(128),
            bndc6 = L.BatchNormalization(64)
        )
        self.train = train
        self.train_dropout = train


    def __call__(self, xi):
        hc0 = F.leaky_relu(self.c0(xi)) 
        hc1 = F.leaky_relu(self.bnc1(self.c1(hc0))) 
        hc2 = F.leaky_relu(self.bnc2(self.c2(hc1))) 
        hc3 = F.leaky_relu(self.bnc3(self.c3(hc2))) 
        hc4 = F.leaky_relu(self.bnc4(self.c4(hc3))) 
        hc5 = F.leaky_relu(self.bnc5(self.c5(hc4))) 
        hc6 = F.leaky_relu(self.bnc6(self.c6(hc5))) 
        hc7 = F.leaky_relu(self.bnc7(self.c7(hc6))) 
        hc8 = F.leaky_relu(self.bnc8(self.c8(hc7))) 

        hc8 = F.expand_dims(hc8,2)      
        h_conv = F.relu(F.dropout(self.bndc00(self.dc00(hc8)), 0.5))


        hc7 = F.expand_dims(hc7,2)
        hc7 = F.broadcast_to(hc7, hc7.data.shape[:2]+(h_conv.data.shape[2],)+hc7.data.shape[3:])
        hc7 = F.concat((h_conv,hc7),1)
        h_conv = F.relu(F.dropout(self.bndc0(self.dc0(hc7)), 0.5))


        hc6 = F.expand_dims(hc6,2)
        hc6 = F.broadcast_to(hc6, hc6.data.shape[:2]+(hc6.data.shape[3],)+hc6.data.shape[3:])
        hc6 = F.concat((h_conv, hc6),1)
        hc6 = F.expand_dims(F.resize_images(F.squeeze(hc6, 0), (16,16), mode='nearest'), 0)
        hc6 = F.dstack([hc6, hc6, hc6, hc6])
        h_conv = F.relu(F.dropout(self.bndc1(self.dc1(hc6)), 0.5))       


        hc5 = F.expand_dims(hc5,2)
        hc5 = F.broadcast_to(hc5, hc5.data.shape[:2]+(hc5.data.shape[3],)+hc5.data.shape[3:])
        hc5 = F.concat((h_conv, hc5),1)
        hc5 = F.expand_dims(F.resize_images(F.squeeze(hc5, 0), (32,32), mode='nearest'), 0)
        h_conv = F.relu(self.bndc2(self.dc2(hc5)))      

       
        hc4 = F.expand_dims(hc4,2)
        hc4 = F.broadcast_to(hc4, hc4.data.shape[:2]+(h_conv.data.shape[2],)+hc4.data.shape[3:])
        hc4 = F.concat((h_conv, hc4),1)
        hc4 = F.expand_dims(F.resize_images(F.squeeze(hc4, 0), (64,64), mode='nearest'), 0)
        h_conv = F.relu(self.bndc3(self.dc3(hc4)))


        hc3 = F.expand_dims(hc3,2)
        hc3 = F.broadcast_to(hc3, hc3.data.shape[:2]+(h_conv.data.shape[2],)+hc3.data.shape[3:])
        hc3 = F.concat((h_conv, hc3),1)
        hc3 = F.expand_dims(F.resize_images(F.squeeze(hc3, 0), (128,128), mode='nearest'), 0)
        h_conv = F.relu(self.bndc4(self.dc4(hc3)))
        
        
        hc2 = F.expand_dims(hc2,2)
        hc2 = F.broadcast_to(hc2, hc2.data.shape[:2]+(h_conv.data.shape[2],)+hc2.data.shape[3:])
        hc2 = F.concat((h_conv, hc2),1)
        hc2 = F.expand_dims(F.resize_images(F.squeeze(hc2, 0), (256,256), mode='nearest'), 0) 
        h_conv = F.relu(self.bndc5(self.dc5(hc2)))
        
        
        hc1 = F.expand_dims(hc1,2) 
        hc1 = F.broadcast_to(hc1, hc1.data.shape[:2]+(h_conv.data.shape[2],)+hc1.data.shape[3:])
        hc1 = F.concat((h_conv,hc1),1)
        h_conv = F.relu(self.bndc6(self.dc6(hc1)))
        
        
        hc0 = F.expand_dims(hc0,2) 
        hc0 = F.broadcast_to(hc0, hc0.data.shape[:2]+(h_conv.data.shape[2],)+hc0.data.shape[3:])
        hc0 = F.concat((h_conv,hc0),1)
        h_conv = self.dc7(hc0)
        

        xi_ = F.expand_dims(xi,2)
        xi_ = F.broadcast_to(xi_, h_conv.data.shape)

        h_conv = F.sigmoid(h_conv+xi_)
        return h_conv
        
        
