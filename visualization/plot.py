import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the uploaded CSV files
allAcc_val = pd.read_csv('./visualization/matterport/allAcc_val.csv')
learning_rate = pd.read_csv('./visualization/matterport/learning_rate.csv')
loss_train = pd.read_csv('./visualization/matterport/loss_train.csv')
loss_train_batch = pd.read_csv('./visualization/matterport/loss_train_batch.csv')
loss_val = pd.read_csv('./visualization/matterport/loss_val.csv')
mAcc_val = pd.read_csv('./visualization/matterport/mAcc_val.csv')
mIoU_val = pd.read_csv('./visualization/matterport/mIoU_val.csv')

def plot_training_graphs():
    # Plot allAcc_val
    plt.figure(figsize=(7, 5))
    sns.lineplot(x=allAcc_val['Step'], y=allAcc_val['Value'])
    plt.title('All Accuracy Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.show()

    # Plot learning_rate
    plt.figure(figsize=(7, 5))
    plt.plot(learning_rate['Step'], learning_rate['Value'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.show()

    # Plot loss_train
    plt.figure(figsize=(7, 5))
    plt.plot(loss_train['Step'], loss_train['Value'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.show()

    # Plot loss_train_batch
    plt.figure(figsize=(7, 5))
    plt.plot(loss_train_batch['Step'], loss_train_batch['Value'])
    plt.title('Training Loss Batch')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.show()

    # Plot loss_val
    plt.figure(figsize=(7, 5))
    plt.plot(loss_val['Step'], loss_val['Value'])
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.show()

    # Plot mAcc_val
    plt.figure(figsize=(7, 5))
    plt.plot(mAcc_val['Step'], mAcc_val['Value'])
    plt.title('Mean Accuracy Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.show()

    # Plot mIoU_val
    plt.figure(figsize=(7, 5))
    plt.plot(mIoU_val['Step'], mIoU_val['Value'])
    plt.title('Mean IoU Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.show()

plot_training_graphs()




