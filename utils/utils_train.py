import os
import torch
import datetime
import torch.nn as nn
import sklearn.metrics as metrics


def train(model, loader_train, loader_valid, epochs, optimizer):
    # create the folders to save weights
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    # the loss function
    criterion = nn.CrossEntropyLoss()

    # save information during training and validation
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # parameter initialization
    train_loss = 0
    best_model = 0

    loss_list_train = []
    loss_list_valid = []

    accuracy_list_train = []
    accuracy_list_valid = []

    # model training
    model.train()

    # cos anneal learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20, T_mult=2)

    for epoch in range(epochs):

        # Initialization accuracy
        train_avg_loss = 0
        valid_avg_loss = 0
        train_accuracy = 0
        valid_accuracy = 0

        # predicted label and real label storage
        train_score_list = []
        train_label_list = []
        valid_score_list = []
        valid_label_list = []

        # Model Training
        model.train()
        for i, (train_images, train_labels) in enumerate(loader_train):
            # split different modal images
            train_img_1, train_img_2, train_img_3 = train_images[0], train_images[1], train_images[2]

            # add to CUDA
            train_img_1, train_img_2 = train_img_1.cuda(), train_img_2.cuda()
            train_img_3, train_labels = train_img_3.cuda(), train_labels.cuda()

            # gradient zero
            optimizer.zero_grad()

            # the loss, uncertain, output (evidence)
            train_evi, loss_ct = model(train_img_1, train_img_2, train_img_3, train_labels, epoch)

            # supervisory losses
            train_loss = criterion(train_evi, train_labels.long()) + loss_ct

            # back propagation
            train_loss.backward()
            optimizer.step()

            # calculate accuracy and average loss
            train_predict = train_evi.detach().max(1)[1]
            train_mid_acc = torch.as_tensor(train_labels == train_predict)
            train_accuracy = train_accuracy + torch.sum(train_mid_acc) / len(train_labels)
            train_avg_loss = train_avg_loss + train_loss / len(loader_train)

            # store predicted and true values
            train_score_list.extend(train_predict.cpu().numpy())
            train_label_list.extend(train_labels.cpu().numpy())

        # updated learning rate
        scheduler.step()

        # Model Testing
        with torch.no_grad():
            model.eval()
            for i, (valid_images, valid_labels) in enumerate(loader_valid):
                # split different modal images
                valid_img_1, valid_img_2, valid_img_3 = valid_images[0], valid_images[1], valid_images[2]

                # add to CUDA
                valid_img_1, valid_img_2 = valid_img_1.cuda(), valid_img_2.cuda()
                valid_img_3, valid_labels = valid_img_3.cuda(), valid_labels.cuda()

                # the loss, uncertain, output (evidence)
                valid_evi, loss_cv = model(valid_img_1, valid_img_2, valid_img_3, valid_labels, epoch)

                # supervisory losses
                valid_loss = criterion(valid_evi, valid_labels.long()) + loss_cv

                #  calculate accuracy and average loss
                valid_predict = valid_evi.detach().max(1)[1]
                valid_mid_acc = torch.as_tensor(valid_labels == valid_predict)
                valid_accuracy = valid_accuracy + torch.sum(valid_mid_acc) / len(valid_labels)
                valid_avg_loss = valid_avg_loss + valid_loss / len(loader_valid)

                # store predicted and true values
                valid_score_list.extend(valid_predict.cpu().numpy())
                valid_label_list.extend(valid_labels.cpu().numpy())

            # recorded losses
            loss_list_train.append(train_loss.detach().cpu().item())
            loss_list_valid.append(valid_loss.detach().cpu().item())

            # recorded accuracy
            accuracy_list_train.append(train_accuracy.detach().cpu().item() / len(loader_train))
            accuracy_list_valid.append(valid_accuracy.detach().cpu().item() / len(loader_valid))

        # accuracy
        train_accuracy_avg = metrics.accuracy_score(train_label_list, train_score_list)
        valid_accuracy_avg = metrics.accuracy_score(valid_label_list, valid_score_list)

        # recall
        train_recall = metrics.recall_score(train_label_list, train_score_list)
        valid_recall = metrics.recall_score(valid_label_list, valid_score_list)

        # f1-score
        train_f1_score = metrics.f1_score(train_label_list, train_score_list)
        valid_f1_score = metrics.f1_score(valid_label_list, valid_score_list)

        # precision
        train_precision = metrics.precision_score(train_label_list, train_score_list)
        valid_precision = metrics.precision_score(valid_label_list, valid_score_list)

        # output the result
        train_avg_loss = train_avg_loss.detach().cpu().item()
        print('Train: Epoch %d, Accuracy %f, Train Loss: %f' % (epoch, train_accuracy_avg, train_avg_loss))

        valid_avg_loss = valid_avg_loss.detach().cpu().item()
        print('Valid: Epoch %d, Accuracy %f, Valid Loss: %f' % (epoch, valid_accuracy_avg, valid_avg_loss))

        # preserve the best validation accuracy model
        if valid_accuracy_avg >= best_model:
            torch.save(model.state_dict(), "save_weights/best_model.pth")
            best_model = valid_accuracy_avg
            print("The current best model has been acquired")

        # record the train_loss, lr, and validation set metrics for each epoch.
        with open(results_file, "a") as f:
            info = f"[epoch: {epoch}]\n" \
                   f"train_loss: {train_avg_loss:.6f}\n" \
                   f"valid_loss: {valid_avg_loss:.6f}\n" \
                   f"train_recall: {train_recall:.4f}\n" \
                   f"valid_recall: {valid_recall:.4f}\n" \
                   f"train_F1_score: {train_f1_score:.4f}\n" \
                   f"valid_F1_score: {valid_f1_score:.4f}\n" \
                   f"train_precision: {train_precision:.4f}\n" \
                   f"valid_precision: {valid_precision:.4f}\n" \
                   f"train_accuracy: {train_accuracy_avg:.6f}\n" \
                   f"valid_accuracy: {valid_accuracy_avg:.6f}\n"
            f.write(info + "\n\n")

    # return to model
    loss = {'Loss1': loss_list_train, 'Loss2': loss_list_valid,
            'Accuracy1': accuracy_list_train, 'Accuracy2': accuracy_list_valid}

    return model, loss
