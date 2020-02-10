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
def bceWithSoftmax(weights=None):
    # i didn't the like the official name of the loss hence the function
    if weights is not None:
        weights = torch.FloatTensor(weights).cuda()
    return torch.nn.CrossEntropyLoss(weights)
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
    # output.backward()