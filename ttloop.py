import torch
import torch.nn.functional as F


def train(
    model,
    dataloader,
    loss_multi,
    loss_bin,
    optimizer,
    device,
    logsoftmax=False,
    task="both"):

    loss_chars, loss_genders = 0, 0

    model.train()
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        char_pred, gender_pred = model(x_batch)

        char_pred_shifted = char_pred[:, :-1, :]
        char_pred_shifted = char_pred_shifted.transpose(1, 2)
        new_x = x_batch[:, 1:]
        if logsoftmax:
            char_pred_shifted = F.log_softmax(char_pred_shifted, dim=2)
        loss_char = loss_multi(char_pred_shifted, new_x)

        gender_pred = gender_pred.type(torch.float32)
        y_batch = y_batch.type(torch.float32)
        loss_gender = loss_bin(gender_pred, y_batch)

        if task == "char":
            loss = loss_char
        elif task == "gender":
            loss = loss_gender
        else:
            loss = loss_char + loss_gender
        loss_chars += loss_char.item() * x_batch.size(0)
        loss_genders += loss_gender.item() * x_batch.size(0)

        loss.backward()
        optimizer.step()

    avg_loss_chars = loss_chars / len(dataloader.dataset)
    avg_loss_genders = loss_genders / len(dataloader.dataset)
    print(
        f"{' ' * 5}Loss(training): characters(multi): {
            avg_loss_chars:.4f} | gender(binary): {avg_loss_genders:.4f}"
    )
    return avg_loss_chars, avg_loss_genders


def test(
    model,
    dataloader,
    loss_multi,
    loss_bin,
    device,
    pad_idx=0,
    roc_auc=None,
    return_all=False,
    logsoftmax=False):

    model.eval()
    t, rocs, count, char_count, acc_char, acc_gender, loss_chars, loss_genders = 0, 0, 0, 0, 0, 0, 0, 0

    all_gender_preds = []
    all_y_targets = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            char_pred, gender_pred = model(x_batch)
            shifted_char_pred = char_pred[:, :-1, :]
            shifted_x = x_batch[:, 1:]

            if roc_auc:
                all_gender_preds.append(gender_pred.cpu())
                all_y_targets.append(y_batch.cpu())

            mask = shifted_x != pad_idx
            predictions = torch.argmax(shifted_char_pred, dim=2)
            correct_chars = predictions == shifted_x
            acc_char += (correct_chars & mask).sum().item()

            acc_gender += ((gender_pred > 0.5) == y_batch).sum().item()
            count += y_batch.size(0)
            char_count += mask.sum().item()

            shifted_char_pred = shifted_char_pred.transpose(1, 2)
            if logsoftmax:
                shifted_char_pred = F.log_softmax(shifted_char_pred, dim=2)
            loss_char = loss_multi(shifted_char_pred, shifted_x)
            gender_pred = gender_pred.float()
            y_batch = y_batch.float()
            loss_gender = loss_bin(gender_pred, y_batch)
            loss_chars += loss_char.item() * x_batch.size(0)
            loss_genders += loss_gender.item() * x_batch.size(0)

    avg_loss_chars = loss_chars / len(dataloader.dataset)
    avg_loss_genders = loss_genders / len(dataloader.dataset)

    avg_char_acc = (acc_char / char_count) * 100
    avg_bin_acc = (acc_gender / count) * 100
    if roc_auc:
        final_preds = torch.cat(all_gender_preds).numpy()
        final_targets = torch.cat(all_y_targets).numpy()
        avg_roc = roc_auc(final_targets, final_preds)

        print(f"{' ' * 5}ROC AUC score (test/avg): {avg_roc:.4f}")
    print(
        f"{' ' * 5}Loss (testing): characters(multi): {
            avg_loss_chars:.4f} | gender(binary): {avg_loss_genders:.4f}"
    )
    print(
        f"{' ' * 5}Accuracy (avg): characters(multi): {
            avg_char_acc:.2f}% | gender(binary): {avg_bin_acc:.2f}%"
    )
    if return_all and roc_auc:
        return (avg_loss_chars, avg_loss_genders), (avg_char_acc, avg_bin_acc), avg_roc
    return avg_loss_chars, avg_loss_genders
