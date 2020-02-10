from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import torch

def get_acc(target,output):
    output_sm = torch.softmax(output, dim=1)
    output_cat = output_sm.argmax(dim=1)
    return accuracy_score(target,output_cat)
def get_recall(target,output,average):
    output_sm = torch.softmax(output, dim=1)
    output_cat = output_sm.argmax(dim=1)
    return recall_score(target,output_cat,average=average)
def conf_mat(target, output, labels=None):
    output_sm = torch.softmax(output, dim=1)
    output_cat = output_sm.argmax(dim=1)
    # print(output_cat)
    mat=confusion_matrix(target, output_cat, labels=labels, sample_weight=None)
    return mat
if __name__=='__main__':
    input = torch.randn(7, 3, requires_grad=True)
    print('input',input)
    target = torch.empty(7, dtype=torch.long).random_(3)
    print('target',target)
    print(conf_mat(target,input))

