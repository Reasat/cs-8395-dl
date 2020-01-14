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
def fl(y,p,alpha=0.7, gamma=2,eps=1e-8):
    y=y.double()
    p=p.double()
    # print(y.max(),p.max())
    # print(torch.log(p+eps),torch.log(torch.relu(1-p)+eps))
    # print((1-p)**gamma,(p)**gamma)
    loss_mat_pos = -(alpha * y * ((1 - p) ** gamma) * torch.log(torch.relu(p) + eps))
    loss_mat_neg = -((1 - alpha) * (1 - y) * (p ** gamma) * torch.log(1 - p + eps))
    loss_mat =  loss_mat_pos+loss_mat_neg
    # print(loss_mat_pos.mean()/loss_mat_neg.mean())
    # print(loss_mat_pos)
    # print(loss_mat_neg)
    # print(loss_mat)
    return loss_mat.sum()
if __name__=='__main__':
    a=torch.tensor([[1,1],[0,0],[0,0]])
    b = torch.tensor([[0.1,0.6], [0.1,0.1], [0.1,0.1]])
    # print(a,b)
    print(fl(a,b,alpha=0.5,gamma=0))