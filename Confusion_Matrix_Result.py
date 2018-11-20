from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def CM_result(y_dev, y_pred,pos=1,report_digits=4):
    accuracy = accuracy_score(y_dev, y_pred)
    precision = precision_score(y_dev, y_pred,pos_label=pos)
    recall = recall_score(y_dev, y_pred,pos_label=pos)
    f1 = f1_score(y_dev, y_pred,pos_label=pos)
    CM_matrix = confusion_matrix(y_dev, y_pred)
    report = classification_report(y_dev, y_pred,digits=report_digits)
    
    print('Accuracy = ' + str(accuracy) +'\nPrecision = ' + str(precision) +'\nRecall = ' + str(recall))
    return accuracy, precision, recall, f1, report, CM_matrix
