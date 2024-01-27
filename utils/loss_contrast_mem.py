from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F



class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, opts, logger):
        super(PixelContrastLoss, self).__init__()

        self.opts = opts
        self.logger = logger
        # self.temperature = self.opts.get('contrast', 'temperature')
        # self.base_temperature = self.opts.get('contrast', 'base_temperature')
        self.temperature = opts.temperature
        self.base_temperature = opts.base_temperature        

        # self.ignore_label = -1
        # if self.opts.exists('loss', 'params') and 'ce_ignore_index' in self.opts.get('loss', 'params'):
        #     self.ignore_label = self.opts.get('loss', 'params')['ce_ignore_index']
        self.ignore_label = [255]
        # self.max_samples = self.opts.get('contrast', 'max_samples')
        # self.max_views = self.opts.get('contrast', 'max_views')
        self.max_samples = 1024
        self.max_views = 50
        self.cache_size = opts.pixel_size + opts.region_size
    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x not in self.ignore_label and x in self.opts.order] # 不考虑unseen class的sample作为锚点
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    self.logger.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _sample_negative_all(self, Q): # 原论文是拿整个memory queue来对比
        # class_num, cache_size, feat_size = Q.shape
        class_num = sum(self.opts.classes[:self.opts.step+1]) # tot_num_classes in current step
        cache_size, feat_size = Q.shape[1], Q.shape[2]
        # order = sorted(self.opts.order) # class index list in current step
        
        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        # for ii in order:
        for ii in range(class_num):
             # if ii == 0: continue  # origin code class 0 means the non-labeled pixels, but 0 means bg class in our task.
            this_q = Q[ii, :cache_size, :]

            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_
    def _sample_negative(self, Q):
        # class_num, cache_size, feat_size = Q.shape
        class_num = sum(self.opts.classes[:self.opts.step+1]) # tot_num_classes in current step
        cache_size, feat_size = Q.shape[1], Q.shape[2]
        # order = sorted(self.opts.order) # class index list in current step
    
        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        # for ii in order:
        for ii in range(class_num):
            # if ii == 0: continue  # origin code class 0 means the non-labeled pixels, but 0 means bg class in our task.
            this_q = Q[ii, :cache_size, :]

            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_
    
    def _contrastive_hard(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]
        y_anchor = torch.cat([torch.full((n_view,), x) for x in y_anchor]).cuda()
        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=0), dim=0)
        
        #正则化特征
        anchor_feature = F.normalize(anchor_feature, dim=1)
        
        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)
            
        #正则化特征
        contrast_feature = F.normalize(contrast_feature, dim=1)

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        logits_hard = []
        y_hard_contrast = []
        for i in range(anchor_num*anchor_count):
            anchor_cls = int(y_anchor[i])
            pos, pos_idx = torch.sort(anchor_dot_contrast[i][self.cache_size*anchor_cls:self.cache_size*(anchor_cls+1)])
            pos = pos[:1024]
            pos_idx = pos_idx[:1024]
            pos_cls = y_contrast[self.cache_size*anchor_cls:self.cache_size*(anchor_cls+1)][pos_idx]
            neg, neg_idx = torch.sort(torch.cat([anchor_dot_contrast[i][:self.cache_size*anchor_cls],anchor_dot_contrast[i][self.cache_size*(anchor_cls+1):]]))
            neg = neg[-2048:]
            neg_idx = neg_idx[-2048:]
            neg_cls = torch.cat([y_contrast[:self.cache_size*anchor_cls],y_contrast[self.cache_size*(anchor_cls+1):]])[neg_idx]
            logits_hard.append(torch.cat([pos,neg]))
            y_hard_contrast.append(torch.cat([pos_cls, neg_cls]).contiguous().view(-1))
        logits_hard = torch.stack(logits_hard)
        y_hard_contrast = torch.stack(y_hard_contrast)
        logits_max, _ = torch.max(logits_hard, dim=1, keepdim=True)
        logits = logits_hard - logits_max.detach()

        # mask = mask.repeat(anchor_count, contrast_count)
        mask = torch.eq(y_anchor, y_hard_contrast).float().cuda()
        neg_mask = 1 - mask

        # logits_mask = torch.ones_like(mask).scatter_(1,
        #                                              torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
        #                                              0)

        # mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def _contrastive_semi_hard(self, X_anchor, y_anchor, queue=None):
        _, n_view = X_anchor.shape[0], X_anchor.shape[1]
        y_anchor = torch.cat([torch.full((n_view,), x) for x in y_anchor]).cuda()
        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=0), dim=0)

        # Normalize feature vectors
        anchor_feature = F.normalize(anchor_feature, dim=1)
        
        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        # Normalize feature vectors
        contrast_feature = F.normalize(contrast_feature, dim=1)
        
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        mask = torch.eq(y_anchor, y_contrast.T).float()
        
        num_instances = anchor_dot_contrast.shape[0]
        
        # Select semi-hard positive and negative samples
        _, sorted_indices = torch.sort(anchor_dot_contrast, dim=1, descending=True)
        sorted_mask = torch.gather(mask, 1, sorted_indices)

        # Extract top 10% nearest negatives and farthest positives
        hard_ratio = 0.1
        hard_negatives = sorted_indices[sorted_mask.bool() & (torch.arange(sorted_mask.shape[1])[None, :] < num_instances * hard_ratio)]
        hard_positives = sorted_indices[~sorted_mask.bool() & (torch.arange(sorted_mask.shape[1])[None, :] > num_instances * (1 - hard_ratio))]

        # Randomly sample from the remaining hard negative and positive indices
        num_neg = self.opts.num_neg
        num_pos = self.opts.num_pos
        sampled_neg_indices = hard_negatives[torch.randint(hard_negatives.shape[0], (num_neg,))]
        sampled_pos_indices = hard_positives[torch.randint(hard_positives.shape[0], (num_pos,))]

        # Compute contrastive loss using the sampled indices
        neg_sim = anchor_dot_contrast[sampled_neg_indices]
        pos_sim = anchor_dot_contrast[sampled_pos_indices]
        
        topk_sim = torch.cat([pos_sim, neg_sim], 1)
        logits_max, _ = torch.max(topk_sim, dim=1, keepdim=True)
        topk_sim = topk_sim - logits_max.detach()
        mask = torch.cat([torch.ones_like(pos_sim), torch.zeros_like(neg_sim)], 1)
        neg_mask = 1 - mask
        
        neg_logits = torch.exp(topk_sim) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(topk_sim)

        log_prob = topk_sim - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]

        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)
        
        #正则化特征
        anchor_feature = F.normalize(anchor_feature, dim=1)
        
        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)
            
        #正则化特征
        contrast_feature = F.normalize(contrast_feature, dim=1)
        
        mask = torch.eq(y_anchor, y_contrast.T).float().cuda()

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)

        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None, queue=None):
        # labels = labels.unsqueeze(1).float().clone()
        # labels = torch.nn.functional.interpolate(labels,
        #                                          (feats.shape[2], feats.shape[3]), mode='nearest')
        # labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive_hard_2(feats_, labels_, queue=queue)
        return loss


class MemContrastLoss(nn.Module, ABC):
    def __init__(self, opts=None, logger = None):
        super(MemContrastLoss, self).__init__()

        self.opts = opts
        self.logger = logger
        # ignore_index = -1
        # if self.opts.exists('loss', 'params') and 'ce_ignore_index' in self.opts.get('loss', 'params'):
        #     ignore_index = self.opts.get('loss', 'params')['ce_ignore_index']

        self.loss_weight = opts.contrast_weight
        self.contrast_criterion = PixelContrastLoss(opts=opts, logger = logger)

    def forward(self, preds, target, cur_epoch=None):
        seg = preds['seg']
        embedding = preds['embed']

        if "segment_queue" in preds:
            segment_queue = preds['segment_queue']
        else:
            segment_queue = None

        if "pixel_queue" in preds:
            pixel_queue = preds['pixel_queue']
        else:
            pixel_queue = None

        if segment_queue is not None and pixel_queue is not None:
            queue = torch.cat((segment_queue, pixel_queue), dim=1)

            _, predict = torch.max(seg, 1)
            loss_contrast = self.contrast_criterion(embedding, target, predict, queue)
        else:
            loss_contrast = 0
        if cur_epoch > self.opts.warm_epoch:
            return self.loss_weight * loss_contrast
        else:
            return 0 * loss_contrast  # just a trick to avoid errors in distributed training

