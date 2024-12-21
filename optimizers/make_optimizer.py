import torch.optim as optim
from torch.optim import lr_scheduler


def make_optimizer(model, opt):
    ignored_params = []
    for i in [model.module.model_uav]:
        ignored_params += list(map(id, i.transformer.parameters()))
    extra_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    base_params = filter(lambda p: id(p) in ignored_params, model.parameters())
    optimizer_ft = optim.AdamW([
        {'params': base_params, 'lr': opt.lr},
        {'params': extra_params, 'lr': opt.lr * opt.NEK_W}],
        weight_decay=5e-4)
    # setup optimizer
    if opt.USE_old_model:
        print("USE_old_model_freeze_blackbone")
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer_ft = optim.AdamW(params, lr=opt.lr, weight_decay=5e-4)


    # optimizer_ft = optim.AdamW(model.parameters(),lr=opt.lr, weight_decay=5e-4)

    # optimizer_ft = optim.SGD(model.parameters(), lr=opt.lr , weight_decay=5e-4, momentum=0.9, nesterov=True)
    # exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=opt.num_epochs, eta_min=5e-6)
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=opt.num_epochs, eta_min=opt.adamw_cos)
    # print(opt.adamw_cos)
    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=opt.steps, gamma=0.5)  # seting lr_rate

    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.9)
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=4, verbose=True,threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)

    return optimizer_ft, exp_lr_scheduler
