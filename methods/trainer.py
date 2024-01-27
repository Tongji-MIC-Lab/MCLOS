import torch
from torch import distributed
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from utils.loss import KnowledgeDistillationLoss, CosineLoss, \
    UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, CosineKnowledgeDistillationLoss
from .segmentation_module import make_model
from modules.classifier import IncrementalClassifier, CosineClassifier, SPNetClassifier
from .utils import get_scheduler, MeanReduction
from pytorch_metric_learning.miners import TripletMarginMiner
from utils.triplet import merge
import numpy as np
import random
from utils.loss_contrast import ContrastLoss
from utils.loss_contrast_mem import MemContrastLoss
CLIP = 10


class Trainer:
    def __init__(self, task, device, logger, opts):
        self.logger = logger
        self.device = device
        self.task = task
        self.opts = opts
        self.opts.classes = self.task.get_all_classes()
        self.opts.order = self.task.order 
        self.novel_classes = self.task.get_n_classes()[-1]
        self.step = task.step

        self.need_model_old = (opts.born_again or opts.mib_kd > 0 or opts.loss_kd > 0 or
                               opts.l2_loss > 0 or opts.l1_loss > 0 or opts.cos_loss > 0)

        self.n_channels = -1  # features size, will be initialized in make model
        self.model = self.make_model()
        self.model = self.model.to(device)

        self.distributed = False
        self.model_old = None

        if opts.fix_bn:
            self.model.fix_bn()

        if opts.bn_momentum is not None:
            self.model.bn_set_momentum(opts.bn_momentum)

        self.initialize(opts)  # initialize model parameters (e.g. perform WI)

        self.born_again = opts.born_again
        self.dist_warm_start = opts.dist_warm_start
        model_old_as_new = opts.born_again or opts.dist_warm_start
        if self.need_model_old:
            self.model_old = self.make_model(is_old=not model_old_as_new)
            # put the old model into distributed memory and freeze it
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.to(device)
            self.model_old.eval()

        # xxx Set up optimizer
        self.train_only_novel = opts.train_only_novel
        params = []
        if not opts.freeze:
            params.append({"params": filter(lambda p: p.requires_grad, self.model.body.parameters())})
        else:
            for par in self.model.body.parameters():
                par.requires_grad = False

        if opts.lr_head != 0:
            params.append({"params": filter(lambda p: p.requires_grad, self.model.head.parameters()),
                           'lr': opts.lr * opts.lr_head})
        else:
            for par in self.model.head.parameters():
                par.requires_grad = False

        if opts.method != "SPN":
            if opts.train_only_novel:
                params.append({"params": filter(lambda p: p.requires_grad, self.model.cls.cls[task.step].parameters()),
                              'lr': opts.lr * opts.lr_cls})
            else:
                params.append({"params": filter(lambda p: p.requires_grad, self.model.cls.parameters()),
                               'lr': opts.lr * opts.lr_cls})

        self.optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
        self.scheduler = get_scheduler(opts, self.optimizer)
        self.logger.debug("Optimizer:\n%s" % self.optimizer)

        reduction = 'none'
        if opts.mib_ce:
            self.criterion = UnbiasedCrossEntropy(old_cl=len(self.task.get_old_labels()),
                                                  ignore_index=255, reduction=reduction)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

        self.reduction = MeanReduction()

        # Feature distillation
        if opts.l2_loss > 0 or opts.cos_loss > 0 or opts.l1_loss > 0:
            assert self.model_old is not None, "Error, model old is None but distillation specified"
            if opts.l2_loss > 0:
                self.feat_loss = opts.l2_loss
                self.feat_criterion = nn.MSELoss()
            elif opts.l1_loss > 0:
                self.feat_loss = opts.l1_loss
                self.feat_criterion = nn.L1Loss()
            elif opts.cos_loss > 0:
                self.feat_loss = opts.cos_loss
                self.feat_criterion = CosineLoss()
        else:
            self.feat_criterion = None

        # Output distillation
        if opts.loss_kd > 0 or opts.mib_kd > 0:
            assert self.model_old is not None, "Error, model old is None but distillation specified"
            if opts.loss_kd > 0:
                if opts.ckd:
                    self.kd_criterion = CosineKnowledgeDistillationLoss(reduction='mean')
                else:
                    self.kd_criterion = KnowledgeDistillationLoss(reduction="mean", alpha=opts.kd_alpha)
                self.kd_loss = opts.loss_kd
            if opts.mib_kd > 0:
                self.kd_loss = opts.mib_kd
                self.kd_criterion = UnbiasedKnowledgeDistillationLoss(reduction="mean")
        else:
            self.kd_criterion = None

        # Body distillation
        if opts.loss_de > 0:
            assert self.model_old is not None, "Error, model old is None but distillation specified"
            self.de_loss = opts.loss_de
            self.de_criterion = nn.MSELoss()
        else:
            self.de_criterion = None

        #Mem Contrast
        if opts.step == 0:
            self.pixel_update_freq = 10
        else:
            self.pixel_update_freq = opts.pixel_update_freq
        self.pixel_size = opts.pixel_size
        self.region_size = opts.region_size
        
    def _dequeue_and_enqueue(self, keys, labels,
                            segment_queue, segment_queue_ptr,
                            pixel_queue, pixel_queue_ptr):
        batch_size = keys.shape[0]
        feat_dim = keys.shape[1]

        # labels = labels[:, ::self.network_stride, ::self.network_stride]

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            # this_label_ids = [x for x in this_label_ids if x > 0]
            this_label_ids = [x for x in this_label_ids if x != 255]

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()

                # segment enqueue and dequeue
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1) #对lb类求特征均值
                ptr = int(segment_queue_ptr[lb])
                segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                segment_queue_ptr[lb] = (segment_queue_ptr[lb] + 1) % self.region_size #将特征均值入队，进入循环队列中

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, self.pixel_update_freq)
                # feat = this_feat[:, perm[:K]]
                idxs_perm = idxs[perm]
                feat = this_feat[:,idxs_perm[:K]].squeeze()
                
                #不加会报错 
                if feat.ndim <= 1:
                    continue
                
                feat = torch.transpose(feat, 0, 1)
                ptr = int(pixel_queue_ptr[lb])

                if ptr + K >= self.pixel_size:
                    pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    pixel_queue_ptr[lb] = 0
                else:
                    pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    #fix bug
                    #pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + 1) % self.pixel_size 
                    pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + K) % self.pixel_size       
    
    def make_model(self, is_old=False):
        classifier, self.n_channels = self.get_classifier(is_old)
        model = make_model(self.opts, classifier)
        return model

    def distribute(self):
        opts = self.opts
        if self.model is not None:
            # Put the model on GPU
            self.distributed = True
            self.model = DistributedDataParallel(self.model, device_ids=[opts.device_id],
                                                 output_device=opts.device_id, find_unused_parameters=False)

    def get_classifier(self, is_old=False):
        # here distinguish methods!
        opts = self.opts
        if opts.method == "SPN":
            classes = self.task.get_old_labels() if is_old else self.task.get_order()
            cls = SPNetClassifier(opts, classes)
            n_feat = cls.channels
        elif opts.method == 'COS':
            n_feat = self.opts.n_feat
            n_classes = self.task.get_n_classes()[:-1] if is_old else self.task.get_n_classes()
            cls = CosineClassifier(n_classes, channels=n_feat)
        else:
            n_feat = self.opts.n_feat
            n_classes = self.task.get_n_classes()[:-1] if is_old else self.task.get_n_classes()
            cls = IncrementalClassifier(n_classes, channels=n_feat)
        return cls, n_feat

    def initialize(self, opts):
        if opts.init_mib and opts.method == "FT":
            device = self.device
            model = self.model.module if self.distributed else self.model

            classifier = model.cls
            imprinting_w = classifier.cls[0].weight[0]
            bkg_bias = classifier.cls[0].bias[0]

            bias_diff = torch.log(torch.FloatTensor([self.task.get_n_classes()[-1] + 1])).to(device)

            new_bias = (bkg_bias - bias_diff)

            classifier.cls[-1].weight.data.copy_(imprinting_w)
            classifier.cls[-1].bias.data.copy_(new_bias)

            classifier.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def warm_up(self, dataset, epochs=1):
        self.warm_up_(dataset, epochs)
        # warm start means make KD after weight imprinting or similar
        if self.dist_warm_start:
            self.model_old.load_state_dict(self.model.state_dict())

    def warm_up_(self, dataset, epochs=1):
        pass

    def cool_down(self, dataset, epochs=1):
        pass

    def train(self, cur_epoch, train_loader, metrics=None, print_int=10, n_iter=1):
        """Train and return epoch loss"""
        if metrics is not None:
            metrics.reset()
        logger = self.logger
        optim = self.optimizer
        scheduler = self.scheduler
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        device = self.device
        model = self.model
        criterion = self.criterion

        epoch_loss = 0.0
        reg_loss = 0.0
        interval_loss = 0.0
        c_loss = 0.0

        model.train()
        if self.opts.freeze and self.opts.bn_momentum == 0:
            model.module.body.eval()
        if self.opts.lr_head == 0 and self.opts.bn_momentum == 0:
            model.module.head.eval()
        cur_step = 0
        for iteration in range(n_iter):
            train_loader.sampler.set_epoch(cur_epoch*n_iter + iteration)  # setup dataloader sampler
            for images, labels in train_loader:

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                rloss = torch.tensor([0.]).to(self.device)

                optim.zero_grad()
                outputs, feat, body = model(images, return_feat=True, return_body=True)

                # xxx Distillation/Regularization Losses
                if self.model_old is not None:
                    outputs_old, feat_old, body_old = self.model_old(images, return_feat=True, return_body=True)
                    if self.kd_criterion is not None:
                        kd_loss = self.kd_loss * self.kd_criterion(outputs, outputs_old)
                        rloss += kd_loss
                    if self.feat_criterion is not None:
                        feat_loss = self.feat_loss * self.feat_criterion(feat, feat_old)
                        rloss += feat_loss
                    if self.de_criterion is not None:
                        de_loss = self.de_loss * self.de_criterion(feat, feat_old)
                        rloss += de_loss


                loss = self.reduction(criterion(outputs, labels), labels)
                
                # # step 0: Embedding Contrastive Losses
                # loss_c = 0.
                # if self.step == 0:
                #     contrast_loss = ContrastLoss(self.opts, logger)
                #     embedding = F.interpolate(feat, size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze_(0)
                #     loss_c = contrast_loss(outputs, embedding, labels, cur_epoch)
                #     if not np.isnan(loss_c.item()):
                #         loss += loss_c
                #         c_loss += loss_c
                #     else:
                #         loss += 0.
                #         c_loss += 0.
                        
                # step 0: Embedding Contrastive Memory Losses
                loss_c = 0.
                # if self.step == 0:
                contrast_loss = MemContrastLoss(self.opts, logger)
                embedding = F.interpolate(feat, size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze_(0)
                preds = dict(seg=outputs, embed=embedding, segment_queue=self.model.module.segment_queue, pixel_queue=self.model.module.pixel_queue)
                loss_c = contrast_loss(preds, labels, cur_epoch)
                with torch.no_grad():
                    self._dequeue_and_enqueue(embedding, labels,
                                            segment_queue=self.model.module.segment_queue,
                                            segment_queue_ptr=self.model.module.segment_queue_ptr,
                                            pixel_queue=self.model.module.pixel_queue,
                                            pixel_queue_ptr=self.model.module.pixel_queue_ptr)
                
                if not np.isnan(loss_c.item()):
                    loss += loss_c
                    c_loss += loss_c
                else:
                    loss += 0.
                    c_loss += 0.
                
                # Since we noted sometimes that rloss spikes, we added a CLIP value to remove it if higher
                if rloss <= CLIP:
                    loss_tot = loss + rloss
                else:
                    print(f"Warning, rloss is {rloss}! Term ignored")
                    loss_tot = loss

                loss_tot.backward()
                optim.step()
                scheduler.step()
                # if loss is NaN, then discard it by set the loss = 0.
                if not np.isnan(loss.item()):
                    epoch_loss += loss.item()
                    reg_loss += rloss.item()
                    interval_loss += loss_tot.item()
                else:
                    epoch_loss += 0.
                    reg_loss += rloss.item()
                    interval_loss += 0.

                _, prediction = outputs.max(dim=1)  # B, H, W
                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                if metrics is not None:
                    metrics.update(labels, prediction)

                cur_step += 1

                if cur_step % print_int == 0:
                    interval_loss = interval_loss / print_int
                    logger.info(f"Epoch {cur_epoch}, Batch {cur_step}/{n_iter}*{len(train_loader)},"
                                f" Loss={interval_loss} RL={reg_loss/print_int} ContrastLoss={c_loss/print_int}]")
                    logger.debug(f"Loss made of: CE {loss}")
                    # visualization
                    if logger is not None:
                        x = cur_epoch * len(train_loader) * n_iter + cur_step
                        logger.add_scalar('Loss', interval_loss, x)
                    interval_loss = 0.0
                    reg_loss = 0.0
                    c_loss = 0.0

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)

        torch.distributed.reduce(epoch_loss, dst=0)
        torch.distributed.reduce(reg_loss, dst=0)

        if distributed.get_rank() == 0:
            epoch_loss = epoch_loss / distributed.get_world_size() / (len(train_loader) * n_iter)
            reg_loss = reg_loss / distributed.get_world_size() / (len(train_loader) * n_iter)

        # collect statistics from multiple processes
        if metrics is not None:
            metrics.synch(device)

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")

        return epoch_loss, reg_loss

    def validate(self, loader, metrics, ret_samples_ids=None, novel=False):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        criterion = self.criterion
        logger = self.logger

        class_loss = 0.0

        ret_samples = []
        with torch.no_grad():
            model.eval()
            for i, (images, labels) in enumerate(loader):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                outputs = model(images)  # B, C, H, W
                if novel:
                    outputs[:, 1:-self.novel_classes] = -float("Inf")

                loss = criterion(outputs, labels).mean()

                class_loss += loss.item()

                _, prediction = outputs.max(dim=1)  # B, H, W
                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

                if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                    ret_samples.append((images[0].detach().cpu().numpy(),
                                        labels[0], prediction[0]))

            # collect statistics from multiple processes
            metrics.synch(device)

            class_loss = torch.tensor(class_loss).to(self.device)

            torch.distributed.reduce(class_loss, dst=0)

            if distributed.get_rank() == 0:
                class_loss = class_loss / distributed.get_world_size() / len(loader)

            if logger is not None:
                logger.info(f"Validation, Class Loss={class_loss}")

        return class_loss, ret_samples

    def load_state_dict(self, checkpoint, strict=True):
        state = {}
        if self.need_model_old or not self.distributed:
            for k, v in checkpoint["model"].items():
                state[k[7:]] = v

        model_state = state if not self.distributed else checkpoint['model']

        if self.born_again and strict:
            self.model_old.load_state_dict(state)
            self.model.load_state_dict(model_state)
        else:
            if self.need_model_old and not strict:
                self.model_old.load_state_dict(state, strict=not self.dist_warm_start)  # we are loading the old model

            if 'module.cls.class_emb' in state and not strict:  # if distributed
                # remove from checkpoint since SPNClassifier is not incremental
                del state['module.cls.class_emb']

            if 'cls.class_emb' in state and not strict:  # if not distributed
                # remove from checkpoint since SPNClassifier is not incremental
                del state['cls.class_emb']

            self.model.load_state_dict(model_state, strict=strict)

            if not self.born_again and strict:  # if strict, we are in ckpt (not step) so load also optim and scheduler
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.scheduler.load_state_dict(checkpoint["scheduler"])

    def state_dict(self):
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict()}
        return state
