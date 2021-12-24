def l2_loss(predictions, targets=None):
    if targets is None:
        return 0.5 * (predictions) ** 2
    else:
        return 0.5 * (predictions - targets) ** 2
