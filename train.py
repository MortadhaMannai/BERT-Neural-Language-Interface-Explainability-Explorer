import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import optim
from tqdm import tqdm


@torch.no_grad()
def evaluate(dataloader, model, device):
    eval_running_loss = 0
    all_pred = []
    all_gt = []

    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        label = batch['label']
        batch.pop('label')
        logits, _ = model(**batch)
        loss = model.calc_loss(logits, label)
        eval_running_loss += loss.item()
        pred = logits.max(-1)[1]

        all_pred.append(pred)
        all_gt.append(label)

    all_pred = torch.cat(all_pred, dim=0).cpu().numpy()
    all_gt = torch.cat(all_gt, dim=0).cpu().numpy()

    accuracy = accuracy_score(all_gt, all_pred)
    eval_loss = eval_running_loss / len(dataloader)
    report = classification_report(
        all_gt, all_pred, target_names=['neutral', 'entailment', 'contradiction'])

    model.train()
    return eval_loss, accuracy, report


def train(dataloaders, model, args, device):
    train_loader, valid_loader, test_loader = dataloaders
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lrdecay)

    val_best_acc = -1
    global_step = 0
    for epoch in range(args.epochs):
        epoch_step = 0
        running_loss = 0
        model.train()
        pbar = tqdm(train_loader)
        for batch in pbar:
            # print(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            label = batch['label']
            batch.pop('label')
            logits, _ = model(**batch)

            optimizer.zero_grad()
            loss = model.calc_loss(logits, label)
            model.backward(loss)
            optimizer.step()

            running_loss += loss.item()
            global_step += 1
            epoch_step += 1
            pbar.set_postfix({'train loss': running_loss / epoch_step})
            if global_step % args.decaystep == 0:
                scheduler.step()

        # evaluation
        valid_loss, valid_acc, val_report = evaluate(valid_loader, model, device)
        test_loss, test_acc, test_report = evaluate(test_loader, model, device)

        print(f'valid loss: {valid_loss}, test loss: {test_loss}')
        print('Validation', val_report)
        print('test', test_report)

        if valid_acc > val_best_acc:
            val_best_acc = valid_acc
            torch.save(model.state_dict(), 'models/veri_net.pt')
