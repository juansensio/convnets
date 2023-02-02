def accuracy(preds, targets):
	return (preds.argmax(dim=1) == targets).float().mean()

def error(preds, targets):
	return 1. - accuracy(preds, targets)

# top k error
# https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
def topk_error(preds, targets, k):
	_, y_pred = preds.topk(k, dim=1)
	y_pred = y_pred.t()
	target_reshaped = targets.view(1, -1).expand_as(y_pred)
	correct = (y_pred == target_reshaped)
	ind_which_topk_matched_truth = correct[:k]
	flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()
	tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)
	return 1. - tot_correct_topk / targets.shape[0]

def top5_error(preds, targets):
	return topk_error(preds, targets, 5)