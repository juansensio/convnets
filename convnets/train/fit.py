import torch
from fastprogress.fastprogress import master_bar, progress_bar
import numpy as np

def fit(
    model, 
    dataloader, 
    optimizer, 
    criterion, 
    metrics,
    device="cpu", 
    epochs=20,
    overfit=0,
    after_epoch_log=True,
    compile=False,
    on_epoch_end=None,
    limit_train_batches=0,
    use_amp = True, 
    after_val=lambda x: None
):
    print("Training model on", device)
    # count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params}")
    if compile:
        print("Compiling model...", end=" ")
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)
        print("Done.")
    model.to(device)
    mb = master_bar(range(1, epochs+1))
    hist = {'epoch': [], 'loss': [], 'lr': []}
    for metric in metrics.keys():
        hist[metric] = []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if not overfit and 'val' in dataloader:
        hist['val_loss'] = []
        for metric in metrics.keys():
            hist['val_' + metric] = []
    for epoch in mb:
        model.train()
        train_logs = {'loss': []}
        for metric in metrics.keys():
            train_logs[metric] = []
        hist['epoch'].append(epoch)
        for batch_ix, batch in enumerate(progress_bar(dataloader['train'], parent=mb)):
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                y_hat = model(X)
                loss = criterion(y_hat, y)
            scaler.scale(loss).backward()
            # gradient clipping
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()
            train_logs['loss'].append(loss.item())
            for metric in metrics.keys():
                train_logs[metric].append(metrics[metric](y_hat, y).item())
            log = f"loss {np.mean(train_logs['loss']):.5f}"
            for metric in metrics.keys():
                log += f" {metric} {np.mean(train_logs[metric]):.5f}"
            mb.child.comment = log
            if overfit and batch_ix > overfit:
                break
            if limit_train_batches and batch_ix > limit_train_batches:
                break
        hist['loss'].append(np.mean(train_logs['loss']))
        for metric in metrics.keys():
            hist[metric].append(np.mean(train_logs[metric]))
        if not overfit and 'val' in dataloader:
            val_logs = {'loss': []}
            for metric in metrics.keys():
                val_logs[metric] = []
            model.eval()
            with torch.no_grad():
                for batch in progress_bar(dataloader['val'], parent=mb):
                    X, y = batch
                    X, y = X.to(device), y.to(device)
                    y_hat = model(X)
                    loss = criterion(y_hat, y)
                    val_logs['loss'] .append(loss.item())
                    for metric in metrics.keys():
                        val_logs[metric].append(metrics[metric](y_hat, y).item())
                    _log = f"val_loss {np.mean(val_logs['loss']):.5f}"
                    for metric in metrics.keys():
                        _log += f" val_{metric} {np.mean(val_logs[metric]):.5f}"
                    mb.child.comment = _log
            hist['val_loss'].append(np.mean(val_logs['loss']))
            for metric in metrics.keys():
                hist['val_' + metric].append(np.mean(val_logs[metric]))
            log += f" val_loss {np.mean(val_logs['loss']):.5f}"
            for metric in metrics.keys():
                log += f" val_{metric} {np.mean(val_logs[metric]):.5f}"
            after_val(val_logs)
        hist['lr'].append(optimizer.param_groups[0]['lr'])
        mb.main_bar.comment = log
        if after_epoch_log: 
            mb.write(f"Epoch {epoch}/{epochs} " + log)
        if on_epoch_end is not None:
            on_epoch_end(hist, model, optimizer, epoch)
    return hist