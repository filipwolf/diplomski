import torch


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation F1 score is less than the previous best ones, then save the
    model state.
    """

    def __init__(self, best_valid_f1=float('inf')):
        self.best_valid_f1 = best_valid_f1

    def __call__(self, current_valid_f1, epoch, model, optimizer, criterion):
        if current_valid_f1 < self.best_valid_f1:
            self.best_valid_f1 = current_valid_f1
            print(f"\nBest validation f1: {self.best_valid_f1}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': criterion,
            }, 'outputs/best_model.pth')
