import torch
from fastprogress.fastprogress import master_bar, progress_bar
import numpy as np

def fit(
    model, 
    dataloader, 
    optimizer, 
    criterion, 
    metrics,
    max_epochs,
    # debug
    overfit_batches=0,
    limit_train_batches=0,
    limit_val_batches=0,
    after_epoch_log=True,
    # callbacks
    on_epoch_end=lambda h,m,o: None,
    after_val=lambda vl: None,
    # device
    device="cpu", 
    rank=0,
    compile=False,
    *args,
    **kwargs
):
    if device == "cuda":
        device_type = "cuda"
        device = f"cuda:{rank}"
    else: 
        device_type = "cpu"
    print("Training model on", device)
    # count parameters
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {n_params}")
    if compile:
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        # torch.set_float32_matmul_precision('high')
        print(rank, "Compiling model ...") # it can takw a while...
        model = torch.compile(model)
    model.to(device)
    mb = master_bar(range(1, max_epochs+1)) if rank == 0 else range(1, max_epochs+1)
    hist = {'epoch': [], 'loss': [], 'lr': []}
    for metric in metrics.keys():
        hist[metric] = []
    if not overfit_batches and 'val' in dataloader:
        hist['val_loss'] = []
        for metric in metrics.keys():
            hist['val_' + metric] = []
    for epoch in mb:
        model.train()
        train_logs = {'loss': []}
        for metric in metrics.keys():
            train_logs[metric] = []
        hist['epoch'].append(epoch)
        pbar = progress_bar(dataloader['train'], parent=mb) if rank == 0 else dataloader['train']
        for batch_ix, batch in enumerate(pbar):
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                y_hat = model(X)
                loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_logs['loss'].append(loss.item())
            for metric in metrics.keys():
                train_logs[metric].append(metrics[metric](y_hat, y).item())
            log = f"loss {np.mean(train_logs['loss']):.5f}"
            for metric in metrics.keys():
                log += f" {metric} {np.mean(train_logs[metric]):.5f}"
            if rank == 0: mb.child.comment = log
            if overfit_batches and batch_ix > overfit_batches:
                break
            if limit_train_batches and batch_ix > limit_train_batches:
                break
        hist['loss'].append(np.mean(train_logs['loss']))
        for metric in metrics.keys():
            hist[metric].append(np.mean(train_logs[metric]))
        if not overfit_batches and 'val' in dataloader and dataloader['val'] is not None:
            val_logs = {'loss': []}
            for metric in metrics.keys():
                val_logs[metric] = []
            model.eval()
            with torch.no_grad():
                pbar = progress_bar(dataloader['val'], parent=mb) if rank == 0 else dataloader['val']
                for batch_ix, batch in enumerate(pbar):
                    X, y = batch
                    X, y = X.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        y_hat = model(X)
                        loss = criterion(y_hat, y)
                    val_logs['loss'] .append(loss.item())
                    for metric in metrics.keys():
                        val_logs[metric].append(metrics[metric](y_hat, y).item())
                    if rank == 0:
                        _log = f"val_loss {np.mean(val_logs['loss']):.5f}"
                        for metric in metrics.keys():
                            _log += f" val_{metric} {np.mean(val_logs[metric]):.5f}"
                        mb.child.comment = _log
                    if limit_val_batches and batch_ix > limit_val_batches:
                        break
            hist['val_loss'].append(np.mean(val_logs['loss']))
            for metric in metrics.keys():
                hist['val_' + metric].append(np.mean(val_logs[metric]))
            if rank == 0:
                log += f" val_loss {np.mean(val_logs['loss']):.5f}"
                for metric in metrics.keys():
                    log += f" val_{metric} {np.mean(val_logs[metric]):.5f}"
            after_val(val_logs)
        hist['lr'].append(optimizer.param_groups[0]['lr'])
        if rank == 0: mb.main_bar.comment = log
        if after_epoch_log and rank == 0: 
            mb.write(f"Epoch {epoch}/{max_epochs} " + log)
        on_epoch_end(hist, model, optimizer)
    return hist