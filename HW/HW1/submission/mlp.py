import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        # self.clear_grad_and_cache()
        self.cache['x'] = x

        W1, b1, W2, b2 = self.parameters['W1'], self.parameters['b1'], self.parameters['W2'], self.parameters['b2']
        z1 = x @ W1.T + b1
        self.cache['z1'] = z1

        if self.f_function == 'relu':
            z2 = torch.nn.ReLU()(z1)
        elif self.f_function == 'sigmoid':
            z2 = torch.nn.Sigmoid()(z1)
        elif self.f_function == 'identity':
            z2 = torch.nn.Identity()(z1)
        else:
            raise Exception('f_function {} is not allowed.'.format(self.f_function))
        self.cache['z2'] = z2

        z3 = z2 @ W2.T + b2
        # z3 = torch.mul(torch.transpose(W2, 0, 1), z2) + b2
        self.cache['z3'] = z3

        if self.g_function == 'relu':
            y_hat = torch.nn.ReLU()(z3)
        elif self.g_function == 'sigmoid':
            y_hat = torch.nn.Sigmoid()(z3)
        elif self.g_function == 'identity':
            y_hat = torch.nn.Identity()(z3)
        else:
            raise Exception('g_function {} is not allowed.'.format(self.f_function))
        self.cache['y_hat'] = y_hat
        # for k, tensor in self.parameters.items():
        #     print(k, tensor.size())

        return y_hat
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        x, z1, z2, z3, y_hat = self.cache['x'], self.cache['z1'], self.cache['z2'], self.cache['z3'], self.cache['y_hat']
        W1, b1, W2, b2 = self.parameters['W1'], self.parameters['b1'], self.parameters['W2'], self.parameters['b2']
        # g
        if self.g_function == 'relu':
            dJdz3 = dJdy_hat.clone() * (z3>0)
        elif self.g_function == 'sigmoid':
            dJdz3 = dJdy_hat * y_hat * (1 - y_hat)
        elif self.g_function == 'identity':
            dJdz3 = dJdy_hat.clone()
        else:
            raise Exception('g_function {} is not allowed.'.format(self.f_function))

        # Linear_2
        dJdW2 = dJdz3.T @ z2
        dJdb2 = dJdz3.T @ torch.ones(dJdz3.shape[0])
        self.grads['dJdb2'] = dJdb2
        self.grads['dJdW2'] = dJdW2

        dJ_dz2 = dJdz3 @ W2

        # f
        if self.f_function == 'relu':
            dJdz1 = dJ_dz2 * (z1 > 0)
        elif self.f_function == 'sigmoid':
            dJdz1 = dJ_dz2 * z2 * (1 - z2)
        elif self.f_function == 'identity':
            dJdz1 = dJ_dz2.clone()
        else:
            raise Exception('f_function {} is not allowed.'.format(self.f_function))

        # Linear_1
        dJdW1 = dJdz1.T @ x
        dJdb1 = dJdz1.T @ torch.ones(dJdz1.shape[0])
        self.grads['dJdb1'] = dJdb1
        self.grads['dJdW1'] = dJdW1

        # self.parameters['W1'] -= self.grads['dJdW1']
        # self.parameters['b1'] -= self.grads['dJdb1']
        # self.parameters['W2'] -= self.grads['dJdW2']
        # self.parameters['b2'] -= self.grads['dJdb2']

        return dJdy_hat

    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    diff = y_hat - y
    loss = (diff @ diff.T).norm()
    # loss = torch.mul(diff, torch.transpose(diff, 0, 1)).norm(dim=1)
    dJdy_hat = (2 * diff) / (y.size()[0] * y.size()[1])

    return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor

    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    loss = - (torch.mul(y, torch.log(y_hat)) + torch.mul(1 - y, torch.log(1-y_hat)))
    dJdy_hat = (- y / y_hat + (1-y) / (1-y_hat)) / (y.size()[0] * y.size()[1])  # why cant i use norm?

    return loss, dJdy_hat

