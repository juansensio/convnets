import torch
from fastprogress.fastprogress import master_bar, progress_bar
import numpy as np

def fit(
    model, 
    dataloader, 
    optimizer, 
    criterion, 
    device="cpu", 
    epochs=20,
    overfit=0,
    log=True,
    compile=False,
    on_epoch_end=None,
    limit_train_batches=0,
):
    print("Training model on", device)
    if compile:
        print("Compiling model...", end=" ")
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)
        print("Done.")
    model.to(device)
    mb = master_bar(range(1, epochs+1))
    hist = {'error': [], 'epoch': [], 'loss': []}
    if not overfit and 'test' in dataloader:
        hist['val_error'] = []
        hist['val_loss'] = []
    for epoch in mb:
        model.train()
        train_loss, train_err = [], []
        hist['epoch'].append(epoch)
        for batch_ix, batch in enumerate(progress_bar(dataloader['train'], parent=mb)):
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
            train_err.append(1. - acc)
            mb.child.comment = f"loss {np.mean(train_loss):.5f} error {np.mean(train_err):.5f}"
            if overfit and batch_ix > overfit:
                break
            if limit_train_batches and batch_ix > limit_train_batches:
                break
        hist['error'].append(np.mean(train_err))
        hist['loss'].append(np.mean(train_loss))
        _log = f"loss {np.mean(train_loss):.5f} error {np.mean(train_err):.5f}"
        if not overfit and 'test' in dataloader:
            val_loss, val_error = [], []
            model.eval()
            with torch.no_grad():
                for batch in progress_bar(dataloader['test'], parent=mb):
                    X, y = batch
                    X, y = X.to(device), y.to(device)
                    y_hat = model(X)
                    loss = criterion(y_hat, y)
                    val_loss.append(loss.item())
                    acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
                    val_error.append(1. - acc)
                    mb.child.comment = f"val_loss {np.mean(val_loss):.5f} val_error {np.mean(val_error):.5f}"
            hist['val_error'].append(np.mean(val_error))
            hist['val_loss'].append(np.mean(val_loss))
            _log += f" val_loss {np.mean(val_loss):.5f} val_error {np.mean(val_error):.5f}"
        mb.main_bar.comment = _log
        if log: 
            mb.write(f"Epoch {epoch}/{epochs} " + _log)
        if on_epoch_end is not None:
            on_epoch_end(hist, model, optimizer, epoch)
    return hist