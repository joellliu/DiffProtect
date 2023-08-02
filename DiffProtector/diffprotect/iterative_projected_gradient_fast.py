# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn

from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import is_float_or_torch_tensor
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import replicate_input
from advertorch.utils import batch_l1_proj

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from advertorch.attacks.utils import rand_init_delta
from advertorch.utils import normalize_by_pnorm


def perturb_iterative(xvar, yvar, xT, predict, nb_iter, eps, eps_iter, loss_fn,
                      T=20, T_enc=50, start_t=20, attack_iter=1, attack_inf_iter=1, repeat_times=1, cnt_skip=0, delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0, alpha=1., lam=0.,
                      l1_sparsity=None, parse=None, parse_map=None, loss_parse_fn=None, vis_full=False, combine=False, target=None):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.

    :param xvar: input data.
    :param yvar: input labels.
    :param predict: forward pass function.
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size.
    :param loss_fn: loss function.
    :param delta_init: (optional) tensor contains the random initialization.
    :param minimize: (optional bool) whether to minimize or maximize the loss.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param l1_sparsity: sparsity value for L1 projection.
                  - if None, then perform regular L1 projection.
                  - if float value, then perform sparse L1 descent from
                    Algorithm 1 in https://arxiv.org/pdf/1904.13000v1.pdf
    :return: tensor containing the perturbed input.
    """
    if delta_init is not None:
        delta = delta_init
        # delta = torch.zeros_like(xT)
        # g = torch.zeros_like(xvar)
    else:
        delta = torch.zeros_like(xvar)
        # delta = torch.zeros_like(xT)
        # g = torch.zeros_like(xvar)
    # print(delta.size())
    # print(xT.size())
    delta.requires_grad_()
    indices = list(range(T))[::-1]  # [T-1, T-2, ..., 0]
    # print(nb_iter, T, start_t)
    # indices = list(range(T_enc))[::-1]  # [T-1, T-2, ..., 0]
    # xT_start = xT.clone().detach()
    # xT_list = xT.clone().detach()
    # xT_start = xT_list[start_t-1].clone().detach()
    # xT = xT_start.clone().detach()

    # alpha = 0.3
    assert start_t <= T
    assert (attack_inf_iter + attack_iter) * repeat_times <= start_t

    if vis_full:
        x_list = []

    for j in range(nb_iter):
        xT_tmp = xT
        # for ii in range(T):
        xT_tmp = xT_tmp.detach()
        # xT = xT.detach()
        # if attack_inf_iter != 0:
        #     for jj in range(attack_inf_iter):
        #         t = torch.tensor(indices[T - start_t + jj] * xT_tmp.size(0), device='cuda').unsqueeze(0)
        #         xT_tmp, x = predict(xT_tmp, xvar + delta, T, t, False, False)
        # else:
        #     jj = 0

        # for ii in range(T):
        cnt = cnt_skip
        for k in range(repeat_times):

            for ii in range(attack_inf_iter + attack_iter):
            # for ii in range(attack_iter):
                # print(ii)
                # t = torch.tensor(indices[ii] * xT.size(0), device='cuda').unsqueeze(0)
                # t = torch.tensor(indices[T - start_t + jj + ii] * xT_tmp.size(0), device='cuda').unsqueeze(0)
                t = torch.tensor(indices[T - start_t + (k * (attack_inf_iter + attack_iter) + ii)] * xT_tmp.size(0), device='cuda').unsqueeze(0)

                # outputs, x_pred, x = predict(xT, xvar + delta, T, t, True)
                # xT_tmp.requires_grad_()
                # outputs, xT_tmp, x = predict(xT_tmp, xvar + delta, T, t, True, False)
                # if indices[ii] >= attack_iter:
                # if ii < attack_inf_iter:
                #     # xT_tmp = xT_tmp.detach()
                #     continue

                if ii < attack_inf_iter:
                    # xT_tmp = xT_tmp.detach()
                    xT_tmp, x = predict(xT_tmp, xvar + delta, T, t, False, False)
                    if vis_full:
                        x_list.append(xT_tmp)
                    continue
                else:
                    xT_tmp.requires_grad_()
                    outputs, xT_tmp, x = predict(xT_tmp, xvar + delta, T, t, True, False)
                    if vis_full:
                        x_list.append(xT_tmp)


                # outputs, xT_tmp, x = predict(xT_tmp + delta, xvar, T, t, True, False)
                # delta_old = delta.data
                # if ii < attack_inf_iter:
                #     # outputs, x_pred, x = predict(xT, xvar + delta, T, t, True, ii == (T-1))
                #     # outputs, xT_tmp, x = predict(xT_tmp, xvar + delta, T, t, True, False)
                #     # xT_tmp.requires_grad_()
                #     xT_tmp, x = predict(xT_tmp, xvar + delta, T, t, False, False)
                #
                #     # xT = x_pred  # No detach!
                #     continue
                # #
                # #
                # else:
                #     if attack_iter > 0:
                #         # x_pred, x = predict(xT, xvar + delta, T, t, False, ii == (T-1))
                #         # xT_tmp, x = predict(xT_tmp, xvar + delta, T, t, False, False)
                #         xT_tmp.requires_grad_()
                #         outputs, xT_tmp, x = predict(xT_tmp.detach(), xvar + delta, T, t, True, False)
                #         attack_iter -= 1
                #         # xT_tmp = xT_tmp.detach()
                #     # else:
                #     #     break
                #         # xT = x_pred  # No detach!


                # outputs.requires_grad_()
                # l1_loss = torch.nn.L1Loss()
                # tgt = torch.zeros_like(outputs).to(outputs.device)
                # loss = l1_loss(outputs, tgt)

                #outputs[0].requires_grad_()

                # l1_loss = torch.nn.L1Loss()
                # pred = outputs[0].detach()
                # tgt = yvar[0].detach()
                # pred.requires_grad_()

                # pred = [output.detach() for output in outputs]
                # pred[0].requires_grad_()
                # tgt = [y.detach() for y in yvar]
                # loss = loss_fn(pred, tgt)


                #loss = l1_loss(pred, tgt)
                #loss = loss_fn([pred], [tgt])
                if parse is not None:
                    parse_map_x = parse(x)[0]
                    if minimize:
                        loss = loss_fn(outputs, yvar) + lam * loss_parse_fn(parse_map_x, parse_map)
                    else:
                        loss = loss_fn(outputs, yvar) - lam * loss_parse_fn(parse_map_x, parse_map)
                    # loss = loss_fn(outputs, yvar)
                    # print(loss_fn(outputs, yvar).item())
                    # print(loss_fn(outputs, yvar).item(), loss_parse_fn(parse_map_x, parse_map).item())
                elif combine and target is not None:
                    loss = loss_fn(outputs, yvar) - loss_fn(outputs, target)
                else:
                    loss = loss_fn(outputs, yvar)

                if minimize:
                    loss = -loss

                loss.backward()
                # loss.backward(retain_graph=True)


                if cnt <= 0:
                    if lam > 0:
                        print(loss_fn(outputs, yvar).item(), loss_parse_fn(parse_map_x, parse_map).item())
                    else:
                        print(loss.item())
                    # g = 1.0 * g + normalize_by_pnorm(
                    #     delta.grad.data, p=1)
                    #
                    # if ord == np.inf:
                    #     delta.data += batch_multiply(eps_iter, torch.sign(g))
                    #     delta.data = batch_clamp(eps, delta.data)
                    #     delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                    #                        ) - xvar.data
                    delta_old = delta.data

                    if ord == np.inf:
                        grad_sign = delta.grad.data.sign()
                        delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
                        delta.data = batch_clamp(eps, delta.data)
                        delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                                           ) - xvar.data

                    elif ord == 2:
                        grad = delta.grad.data
                        grad = normalize_by_pnorm(grad)
                        delta.data = delta.data + batch_multiply(eps_iter, grad)
                        delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                                           ) - xvar.data
                        if eps is not None:
                            delta.data = clamp_by_pnorm(delta.data, ord, eps)

                    elif ord == 1:
                        grad = delta.grad.data
                        abs_grad = torch.abs(grad)

                        batch_size = grad.size(0)
                        view = abs_grad.view(batch_size, -1)
                        view_size = view.size(1)
                        if l1_sparsity is None:
                            vals, idx = view.topk(1)
                        else:
                            vals, idx = view.topk(
                                int(np.round((1 - l1_sparsity) * view_size)))

                        out = torch.zeros_like(view).scatter_(1, idx, vals)
                        out = out.view_as(grad)
                        grad = grad.sign() * (out > 0).float()
                        grad = normalize_by_pnorm(grad, p=1)
                        delta.data = delta.data + batch_multiply(eps_iter, grad)

                        delta.data = batch_l1_proj(delta.data.cpu(), eps)
                        delta.data = delta.data.to(xvar.device)
                        delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                                           ) - xvar.data
                    else:
                        error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
                        raise NotImplementedError(error)

                    delta.data = alpha * delta.data + (1 - alpha) * delta_old

                cnt -= 1

                delta.grad.data.zero_()

                # xT = predict.encode_all(xvar + delta, x, T)
                # xT = xT[start_t * (T_enc // T) - 1]
                # xT = xT.detach()
                xT_tmp = xT_tmp.detach()
                # if ii != (T-1):
                #     xT = x_pred
            # xT = xT_start

    x_adv = clamp(xvar + delta, clip_min, clip_max)
    if vis_full:
        return x_adv, x, x_list
    return x_adv, x


class PGDAttack(Attack, LabelMixin):
    """
    The projected gradient descent attack (Madry et al, 2017).
    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point.
    Paper: https://arxiv.org/pdf/1706.06083.pdf

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, loss_parse_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, l1_sparsity=None, targeted=False, parse=None, T_enc=50, T_atk=20,
            start_t=20, attack_iter=1, attack_inf_iter=1, repeat_times=1, cnt_skip=0, lam=0., vis_full=False, combine=False):
        """
        Create an instance of the PGDAttack.

        """
        super(PGDAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        self.loss_parse_fn = loss_parse_fn
        self.T_enc = T_enc
        self.T_atk = T_atk
        self.start_t = start_t
        self.attack_iter = attack_iter
        self.attack_inf_iter = attack_inf_iter
        self.repeat_times = repeat_times
        self.cnt_skip = cnt_skip
        self.lam = lam
        self.vis_full = vis_full
        self.combine = combine

        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        if self.loss_parse_fn is None:
            self.loss_parse_fn = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
        self.l1_sparsity = l1_sparsity
        self.parse = parse
        # self.parse_map = parse_map
        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)

    def perturb(self, z, y, x, parse_map=None, target=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        
        #x, y = self._verify_and_process_inputs(x, y)
        #print("my pgd")
        delta = torch.zeros_like(z)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_delta(
                delta, z, self.ord, self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(
                z + delta.data, min=self.clip_min, max=self.clip_max) - z

        rval = perturb_iterative(
            z, y, x, self.predict, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,
            l1_sparsity=self.l1_sparsity, T=self.T_atk, parse=self.parse, parse_map=parse_map,
            loss_parse_fn=self.loss_parse_fn, T_enc=self.T_enc, start_t=self.start_t, attack_iter=self.attack_iter,
            attack_inf_iter=self.attack_inf_iter, repeat_times=self.repeat_times, cnt_skip=self.cnt_skip, lam=self.lam, vis_full=self.vis_full,
            combine=self.combine, target=target
        )
        if self.vis_full:
            return rval[0].data, rval[1].data, rval[2]
        else:
            return rval[0].data, rval[1].data


class LinfPGDAttack(PGDAttack):
    """
    PGD Attack with order=Linf

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False):
        ord = np.inf
        super(LinfPGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted,
            ord=ord)


class L2PGDAttack(PGDAttack):
    """
    PGD Attack with order=L2

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False):
        ord = 2
        super(L2PGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted,
            ord=ord)


class L1PGDAttack(PGDAttack):
    """
    PGD Attack with order=L1

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=10., nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False):
        ord = 1
        super(L1PGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted,
            ord=ord, l1_sparsity=None)


class SparseL1DescentAttack(PGDAttack):
    """
    SparseL1Descent Attack

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    :param l1_sparsity: proportion of zeros in gradient updates
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=False, clip_min=0., clip_max=1.,
            l1_sparsity=0.95, targeted=False):
        ord = 1
        super(SparseL1DescentAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted,
            ord=ord, l1_sparsity=l1_sparsity)


class L2BasicIterativeAttack(PGDAttack):
    """Like GradientAttack but with several steps for each epsilon.

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(self, predict, loss_fn=None, eps=0.1, nb_iter=10,
                 eps_iter=0.05, clip_min=0., clip_max=1., targeted=False):
        ord = 2
        rand_init = False
        l1_sparsity = None
        super(L2BasicIterativeAttack, self).__init__(
            predict, loss_fn, eps, nb_iter, eps_iter, rand_init,
            clip_min, clip_max, ord, l1_sparsity, targeted)


class LinfBasicIterativeAttack(PGDAttack):
    """
    Like GradientSignAttack but with several steps for each epsilon.
    Aka Basic Iterative Attack.
    Paper: https://arxiv.org/pdf/1611.01236.pdf

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(self, predict, loss_fn=None, eps=0.1, nb_iter=10,
                 eps_iter=0.05, clip_min=0., clip_max=1., targeted=False):
        ord = np.inf
        rand_init = False
        l1_sparsity = None
        super(LinfBasicIterativeAttack, self).__init__(
            predict, loss_fn, eps, nb_iter, eps_iter, rand_init,
            clip_min, clip_max, ord, l1_sparsity, targeted)


class MomentumIterativeAttack(Attack, LabelMixin):
    """
    The Momentum Iterative Attack (Dong et al. 2017).

    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point. The optimization is performed with
    momentum.
    Paper: https://arxiv.org/pdf/1710.06081.pdf

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations
    :param decay_factor: momentum decay factor.
    :param eps_iter: attack step size.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    :param ord: the order of maximum distortion (inf or 2).
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40, decay_factor=1.,
            eps_iter=0.01, clip_min=0., clip_max=1., targeted=False,
            ord=np.inf):
        """Create an instance of the MomentumIterativeAttack."""
        super(MomentumIterativeAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.decay_factor = decay_factor
        self.eps_iter = eps_iter
        self.targeted = targeted
        self.ord = ord
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        g = torch.zeros_like(x)

        delta = nn.Parameter(delta)

        for i in range(self.nb_iter):

            if delta.grad is not None:
                delta.grad.detach_()
                delta.grad.zero_()

            imgadv = x + delta
            outputs = self.predict(imgadv)
            loss = self.loss_fn(outputs, y)
            if self.targeted:
                loss = -loss
            loss.backward()

            g = self.decay_factor * g + normalize_by_pnorm(
                delta.grad.data, p=1)
            # according to the paper it should be .sum(), but in their
            #   implementations (both cleverhans and the link from the paper)
            #   it is .mean(), but actually it shouldn't matter
            if self.ord == np.inf:
                delta.data += batch_multiply(self.eps_iter, torch.sign(g))
                delta.data = batch_clamp(self.eps, delta.data)
                delta.data = clamp(
                    x + delta.data, min=self.clip_min, max=self.clip_max) - x
            elif self.ord == 2:
                delta.data += self.eps_iter * normalize_by_pnorm(g, p=2)
                delta.data *= clamp(
                    (self.eps * normalize_by_pnorm(delta.data, p=2) /
                        delta.data),
                    max=1.)
                delta.data = clamp(
                    x + delta.data, min=self.clip_min, max=self.clip_max) - x
            else:
                error = "Only ord = inf and ord = 2 have been implemented"
                raise NotImplementedError(error)

        rval = x + delta.data
        return rval


class L2MomentumIterativeAttack(MomentumIterativeAttack):
    """
    The L2 Momentum Iterative Attack
    Paper: https://arxiv.org/pdf/1710.06081.pdf

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations
    :param decay_factor: momentum decay factor.
    :param eps_iter: attack step size.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40, decay_factor=1.,
            eps_iter=0.01, clip_min=0., clip_max=1., targeted=False):
        """Create an instance of the MomentumIterativeAttack."""
        ord = 2
        super(L2MomentumIterativeAttack, self).__init__(
            predict, loss_fn, eps, nb_iter, decay_factor,
            eps_iter, clip_min, clip_max, targeted, ord)


class LinfMomentumIterativeAttack(MomentumIterativeAttack):
    """
    The Linf Momentum Iterative Attack
    Paper: https://arxiv.org/pdf/1710.06081.pdf

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations
    :param decay_factor: momentum decay factor.
    :param eps_iter: attack step size.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40, decay_factor=1.,
            eps_iter=0.01, clip_min=0., clip_max=1., targeted=False):
        """Create an instance of the MomentumIterativeAttack."""
        ord = np.inf
        super(LinfMomentumIterativeAttack, self).__init__(
            predict, loss_fn, eps, nb_iter, decay_factor,
            eps_iter, clip_min, clip_max, targeted, ord)


class FastFeatureAttack(Attack):
    """
    Fast attack against a target internal representation of a model using
    gradient descent (Sabour et al. 2016).
    Paper: https://arxiv.org/abs/1511.05122

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param eps_iter: attack step size.
    :param nb_iter: number of iterations
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    """

    def __init__(self, predict, loss_fn=None, eps=0.3, eps_iter=0.05,
                 nb_iter=10, rand_init=True, clip_min=0., clip_max=1.):
        """Create an instance of the FastFeatureAttack."""
        super(FastFeatureAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.rand_init = rand_init
        self.clip_min = clip_min
        self.clip_max = clip_max
        if self.loss_fn is None:
            self.loss_fn = nn.MSELoss(reduction="sum")

    def perturb(self, source, guide, delta=None):
        """
        Given source, returns their adversarial counterparts
        with representations close to that of the guide.

        :param source: input tensor which we want to perturb.
        :param guide: targeted input.
        :param delta: tensor contains the random initialization.
        :return: tensor containing perturbed inputs.
        """
        # Initialization
        if delta is None:
            delta = torch.zeros_like(source)
            if self.rand_init:
                delta = delta.uniform_(-self.eps, self.eps)
        else:
            delta = delta.detach()

        delta.requires_grad_()

        source = replicate_input(source)
        guide = replicate_input(guide)
        guide_ftr = self.predict(guide).detach()

        xadv = perturb_iterative(source, guide_ftr, self.predict,
                                 self.nb_iter, eps_iter=self.eps_iter,
                                 loss_fn=self.loss_fn, minimize=True,
                                 ord=np.inf, eps=self.eps,
                                 clip_min=self.clip_min,
                                 clip_max=self.clip_max,
                                 delta_init=delta)

        xadv = clamp(xadv, self.clip_min, self.clip_max)

        return xadv.data
