import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset

def mIoU_of_class(prediction, predict_label, target, target_label):
  target_args = torch.where(target == target_label)
  target_size = target_args[0].shape[0]
  if target_size == 0:
    return None
  else:
    intersection = prediction[target_args] == predict_label
    intersection = torch.sum(intersection)
    predict_args = torch.where(prediction == predict_label)
    predict_size = predict_args[0].shape[0]
    union = target_size + predict_size - intersection
    mIoU = intersection.float()/union.float()
    return mIoU

# test_classes is search space starting from 0.
# 0 is background class
def IoU_per_class(model, test_features, test_labels, word_vectors, test_classes, test_batch, GPU, calibrate_classes, calibrate): 
  if (test_classes < 0).any():
    return False

  test_data = TensorDataset(test_features, test_labels)
  test_loader = DataLoader(test_data,batch_size=test_batch,shuffle=False)
  test_size = test_features.shape[0]

  test_classes = np.sort(test_classes)
  if test_classes[0] == 0:
    includeBack = True
  else:
    includeBack = False
  class_num = len(test_classes)
  test_vectors = torch.tensor([word_vectors[int(c-1)] for c in test_classes if c > 0]).view(-1, word_vectors.shape[1]).float().cuda(GPU) # -1*300

  class_acc = [None] * class_num
  predict_total = None

  for batch_features, batch_labels in test_loader:
    batch_size = batch_features.shape[0]
    support_features = test_vectors.repeat(batch_size,1) # -1*300*1*1 -> -1*256*28*28

    query_features = batch_features.repeat(1,test_vectors.shape[0],1,1).view(-1,3,224,224)
    query_features = query_features.cuda(GPU).float()

    relations = model(query_features,support_features).view(batch_size,test_vectors.shape[0],224,224) # -1*-1*224*224

    if includeBack:
      background_scores = 1-torch.max(relations,1)[0].view(-1,1,224,224)
      scores = torch.cat((background_scores,relations),1)
    else:
      scores = relations

    if calibrate_classes is not None and len(calibrate_classes) > 0:
      scores[:,calibrate_classes,:,:] = scores[:,calibrate_classes,:,:] * calibrate

    prediction = torch.max(scores,1)[1]

    if predict_total is None:
      predict_total = prediction.cpu().detach()
    else:
      predict_total = torch.cat((predict_total,prediction.cpu().detach()))

  # ignore unselected classes
  test_labels = test_labels.view(-1,224,224)
  select_args = np.where(np.isin(test_labels,test_classes))
  test_labels = test_labels[select_args]
  predict_total = predict_total[select_args]

  for c_i in range(class_num):
    mIoU = mIoU_of_class(predict_total, c_i, test_labels, test_classes[c_i])
    if mIoU is not None:
      class_acc[c_i] = mIoU.item()
  return class_acc