from ssl_app.core.loss_functions import mse_loss_restoration, mse_loss_registration, ncc_loss, gradient_loss


def get_loss_functions(settings):

    loss_restoration = mse_loss_restoration
    loss_restoration.__name__ = settings['loss']['restoration_loss_name']

    loss_registration = ncc_loss
    loss_registration.__name__ = settings['loss']['registration_loss_name']

    loss_flow_regularization = gradient_loss
    loss_flow_regularization.__name__ = settings['loss']['flow_regularization_loss_name']

    return [loss_restoration, loss_registration, loss_flow_regularization]
