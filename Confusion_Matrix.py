import matplotlib.pyplot as plt
import itertools
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def CM_plot(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize = 50)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize = 40)
    plt.yticks(tick_marks, classes,fontsize = 40)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",fontsize = 60,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize = 40)
    plt.xlabel('Predicted label',fontsize = 40)

def CM_result(y_dev, y_pred,pos=1,report_digits=4):
    accuracy = accuracy_score(y_dev, y_pred)
    precision = precision_score(y_dev, y_pred,pos_label=pos)
    recall = recall_score(y_dev, y_pred,pos_label=pos)
    f1 = f1_score(y_dev, y_pred,pos_label=pos)
    CM_matrix = confusion_matrix(y_dev, y_pred)
    report = classification_report(y_dev, y_pred,digits=report_digits)
    
    print('Accuracy = ' + str(accuracy) +'\nPrecision = ' + str(precision) +'\nRecall = ' + str(recall))
    return accuracy, precision, recall, f1, report, CM_matrix