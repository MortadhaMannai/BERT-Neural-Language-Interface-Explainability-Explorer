def freeze(*modules):
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False


def unfreeze(*modules):
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True
