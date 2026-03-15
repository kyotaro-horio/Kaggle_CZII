import torch

def calc_fbeta(
        y_pred, y_true, #BCDWH 
        beta=4, 
        weights=[1, 0, 2, 1, 2, 1], 
        smooth=1e-7):
    
    batch_size = y_pred.shape[0] 
    n_pred_ch = y_pred.shape[1]

    if n_pred_ch - 1 != len(weights):
        raise AssertionError(
            f'number of channel should be {len(weights) + 1} but input is {n_pred_ch}')
    
    #not include background
    y_pred = y_pred[:, 1:]
    y_true = y_true[:, 1:]

    # Flatten predictions and ground truths
    y_pred = y_pred.contiguous().view(y_pred.shape[0], y_pred.shape[1], -1)
    y_true = y_true.contiguous().view(y_true.shape[0], y_true.shape[1], -1)

    # Compute true positives, false positives, and false negatives
    tp = torch.sum(y_true * y_pred, dim=2)
    fp = torch.sum((1 - y_true) * y_pred, dim=2)
    fn = torch.sum(y_true * (1 - y_pred), dim=2)

    # Compute precision and recall
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    # Compute F-beta score
    beta_sq = beta ** 2
    fbeta_each_class = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + smooth)
    fbeta_each_class_mean = torch.mean(fbeta_each_class, 0)

    lb_score = torch.sum(fbeta_each_class * torch.Tensor(weights).cuda()) / sum(weights) / batch_size
    
    return fbeta_each_class_mean, lb_score

        
if __name__=='__main__':

    p = [[
            [1, 0], 
            [1, 1]
        ], 
        [
            [1, 0], 
            [0, 0]
        ], 
        [
            [0, 1], 
            [0, 1]
        ]]

    t = [[
            [1, 0], 
            [0, 1]
        ], 
        [
            [1, 0], 
            [0, 1]
        ], 
        [
            [0, 1], 
            [0, 0]
        ]]
    
    p = torch.Tensor(p).cuda()
    t = torch.Tensor(t).cuda()

    p = torch.stack([p,p], 0)
    t = torch.stack([t,t], 0)

    print('shape of p:', p.shape)
    print('shape of t:', t.shape)

    f, lb = calc_fbeta_metric_for_czii(p, t, beta=4, weights=[1,2], smooth=1e-7)
    print('mean fbeta of each class:', f)
    print('lb:', lb)

    print()
    print('-- demo calc mean among val steps')
    print(torch.mean(torch.stack([f, f], 0), 0))
    print(torch.mean(torch.stack([lb, lb], 0), 0))