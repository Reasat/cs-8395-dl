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
if __name__=='__main__':
    a=torch.tensor([[1,0],[0,0],[0,0]])
    b = torch.tensor([[0.1,1], [0,0], [0,0]])
    # print(a,b)
    print(fl(a,b,alpha=2,beta=0))