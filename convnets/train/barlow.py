import torch
from fastprogress.fastprogress import master_bar, progress_bar
import numpy as np

def barlow_step(model, batch, device, l=5e-3):
    # two randomly augmented versions of x
    x1, x2 = batch
    x1, x2 = x1.to(device), x2.to(device)

    # compute representations
    z1 = model(x1)
    z2 = model(x2)

    # normalize repr. along the batch dimension
    N, D = z1.shape
    z1_norm = (z1 - z1.mean(0)) / z1.std(0) # NxD
    z2_norm = (z2 - z2.mean(0)) / z2.std(0) # NxD

    # cross-correlation matrix
    c = (z1_norm.T @ z2_norm) / N # DxD

    # loss
    c_diff = (c - torch.eye(D, device=device)).pow(2) # DxD
    # multiply off-diagonal elems of c_diff by lambda
    d = torch.eye(D, dtype=bool)
    c_diff[~d] *= l
    return c_diff.sum()

def barlow_fit(model, dataloader, optimizer, scheduler=None, use_amp = True, epochs=10, device="cuda", log=True, eval_each=10, limit_train_batches=0, ssl_eval=None):
    print("Training model on", device)
    model.to(device)
    mb = master_bar(range(1, epochs+1))
    hist = {'loss': [], 'epoch': []}
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in mb:
        # train
        model.train()
        train_loss = []
        hist['epoch'].append(epoch)
        for batch_ix, batch in enumerate(progress_bar(dataloader, parent=mb)):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = barlow_step(model, batch, device)
            scaler.scale(loss).backward()
            # gradient clipping
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()
            train_loss.append(loss.item())
            mb.child.comment = f"loss {np.mean(train_loss):.5f}"
            if limit_train_batches and batch_ix > limit_train_batches:
                break
        hist['loss'].append(np.mean(train_loss))
        _log = f"loss {np.mean(train_loss):.5f}"
        scheduler.step()
        # eval
        if ssl_eval is not None and not epoch % eval_each:
            print("evaluating ...")
            ssl_eval(model)
        mb.main_bar.comment = _log
        if log: 
            mb.write(f"Epoch {epoch}/{epochs} " + _log)
        torch.save(model.state_dict(), f"./checkpoints/barlow.pth")
    return hist