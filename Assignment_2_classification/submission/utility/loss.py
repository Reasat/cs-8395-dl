import torch

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