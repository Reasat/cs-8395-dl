import torch

def loss_l2(y,y_p, gamma=0):
    #calculate loss per sample
    # loss=torch.FloatTensor([0])
    # for y_s,y_s_p in zip(y,y_p):
    #     loss_s=((y_s - y_s_p) * (y_s - y_s_p)).sum()
    #     loss+=loss_s
    # loss = loss/y.shape[0]
    # overall loss
    y=y.double()
    y_p=y_p.double()
    loss = (((y-y_p)**gamma)*((y-y_p)*(y-y_p))).sum()/y.shape[0]
    return loss
def fl(y,p,alpha=2, beta=4, eps=1e-8, agg='mean'):
    y=y.double()
    p=p.double()
    pos_mask = (y==1).double()
    neg_mask= (y<1).double()
    # print(pos_mask.sum(),neg_mask.sum(),y.shape)
    neg_weights=torch.pow(1-y,beta)
    poss_loss = -torch.log(torch.clamp(p,eps,1))*torch.pow(1-p,alpha)*pos_mask
    neg_loss = -torch.log(torch.clamp(1-p, eps, 1)) * torch.pow(p, alpha) * neg_weights* neg_mask
    # print(poss_loss, neg_loss)
    # print(poss_loss.sum(),neg_loss.sum())
    loss_mat = poss_loss+neg_loss
    if agg=='sum':
        return loss_mat.sum()
    else:
        return loss_mat.mean()

class FTL():
    def __init__(self,smooth=1,alpha=0.7,gamma=4/3):
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma
    def tversky(self,y_true, y_pred):
        y_true_pos = y_true.view(-1)
        y_pred_pos = y_pred.view(-1)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        # print(y_pred_pos)
        # print((y_pred_pos==0).float())
        false_neg = torch.sum(y_true_pos * (1-y_pred_pos))
        false_pos = torch.sum((1-y_true_pos)*y_pred_pos)
        # print(true_pos,false_neg,false_pos)
        return (true_pos + self.smooth)/(true_pos + self.alpha*false_neg + (1-self.alpha)*false_pos + self.smooth)

    def tversky_loss(self,y_true, y_pred):
        return 1 - self.tversky(y_true,y_pred)

    def focal_tversky(self,y_true,y_pred):
        # print(self.tversky(y_true, y_pred))
        return torch.pow((1-self.tversky(y_true, y_pred)), 1/self.gamma)

# pred=torch.Tensor([[[1,0],[1,0]],[[1,1],[1,0]]])
# gt=torch.Tensor([[[1,1],[1,0]],[[1,1],[1,0]]])
# ftl = FTL(smooth=1,alpha=0.7,gamma=1)
# print(ftl.focal_tversky(gt,pred))
# print(pred.shape,pred.view(-1))
def bceWithSoftmax(weights=None):
    # i didn't the like the official name of the loss hence the function
    if weights is not None:
        weights = torch.FloatTensor(weights).cuda()
    return torch.nn.CrossEntropyLoss(weights)

def dice_loss(input, target):
    smooth = 1.
    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()

    return 1-((2. * intersection + smooth) /
            (iflat.sum() + tflat.sum() + smooth))

if __name__=='__main__':
    loss = bceWithSoftmax()
    input = torch.randn(2, 3, requires_grad=True)
    target = torch.empty(2, dtype=torch.long).random_(3)
    print('input',input)
    print('target',target)
    input_sm = torch.softmax(input,dim=1)
    print('softmax',input_sm)
    print(-torch.log(input_sm))
    print(-torch.log(1-input_sm))

    output = loss(input, target)
    print(output)
    # output_batch.backward()