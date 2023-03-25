import torch
# from rich.progress import track
from fastprogress.fastprogress import master_bar, progress_bar
import numpy as np
import lightning as L

# https://lightning.ai/docs/fabric/stable/advanced/distributed_communication.html

def fit(
    model, 
    dataloader, 
    optimizer, 
    criterion, 
    metrics,
    max_epochs,
    compile=True,
    fabric=None,
    # debug
    overfit_batches=0,
    limit_train_batches=0,
    limit_val_batches=0,
    *args,
    **kwargs
):
    fabric = L.Fabric(accelerator="gpu", devices=1, precision='bf16-mixed') if fabric is None else fabric
    fabric.launch()
    # count parameters
    if fabric.global_rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {n_params}")
    if compile:
        # torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        # torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        # torch.set_float32_matmul_precision('high')
        print("Compiling model ...") # it can takw a while...
        model = torch.compile(model)
    hist = {'epoch': [], 'loss': [], 'lr': []}
    for metric in metrics.keys():
        hist[metric] = []
    dataloader['train'] = fabric.setup_dataloaders(dataloader['train'])
    if not overfit_batches and 'val' in dataloader:
        dataloader['val'] = fabric.setup_dataloaders(dataloader['val'])
        hist['val_loss'] = []
        for metric in metrics.keys():
            hist['val_' + metric] = []
    model, optimizer = fabric.setup(model, optimizer)
    fabric.call('before_start')
    mb = master_bar(range(1, max_epochs+1)) if fabric.global_rank == 0 else range(1, max_epochs+1)
    for epoch in mb:
    # for epoch in range(1, max_epochs+1):
        model.train()
        train_logs = {'loss': []}
        for metric in metrics.keys():
            train_logs[metric] = []
        hist['epoch'].append(epoch)
        # pbar = tqdm(dataloader['train'])
        pbar = progress_bar(dataloader['train'], parent=mb) if fabric.global_rank == 0 else dataloader['train']
        for batch_ix, batch in enumerate(pbar):
        # for batch_ix, batch in track(enumerate(dataloader['train']), total=len(dataloader['train']), description=f"Epoch {epoch}/{max_epochs}"):
            X, y = batch
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            fabric.backward(loss)
            # loss.backward()
            optimizer.step()
            train_logs['loss'].append(loss.item())
            for metric in metrics.keys():
                train_logs[metric].append(metrics[metric](y_hat, y).item())
            log = f"loss {np.mean(train_logs['loss']):.5f}"
            for metric in metrics.keys():
                log += f" {metric} {np.mean(train_logs[metric]):.5f}"
            if fabric.global_rank == 0: mb.child.comment = log
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
                pbar = progress_bar(dataloader['val'], parent=mb) if fabric.global_rank == 0 else dataloader['val']
                for batch_ix, batch in enumerate(pbar):
                    X, y = batch
                    y_hat = model(X)
                    loss = criterion(y_hat, y)
                    val_logs['loss'] .append(loss.item())
                    for metric in metrics.keys():
                        val_logs[metric].append(metrics[metric](y_hat, y).item())
                    if fabric.global_rank == 0:
                        _log = f"val_loss {np.mean(val_logs['loss']):.5f}"
                        for metric in metrics.keys():
                            _log += f" val_{metric} {np.mean(val_logs[metric]):.5f}"
                        mb.child.comment = _log
                    if limit_val_batches and batch_ix > limit_val_batches:
                        break
            hist['val_loss'].append(np.mean(val_logs['loss']))
            for metric in metrics.keys():
                hist['val_' + metric].append(np.mean(val_logs[metric]))
            if fabric.global_rank == 0:
                log += f" val_loss {np.mean(val_logs['loss']):.5f}"
                for metric in metrics.keys():
                    log += f" val_{metric} {np.mean(val_logs[metric]):.5f}"
            fabric.call("after_val", val_logs=val_logs)
        hist['lr'].append(optimizer.param_groups[0]['lr'])
        if fabric.global_rank == 0:
            mb.main_bar.comment = log
            mb.write(f"Epoch {epoch}/{max_epochs} " + log)
        # fabric.log_dict({k: v[-1] for k, v in hist.items()})
        fabric.call("after_epoch", hist=hist)
    return hist