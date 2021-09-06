from torch import nn
import torch.nn.functional as F
import torch
from data.const import ENTITY_PADDING, RELATION_PADDING, task_ner_labels
from utils.utils import accuracy



class MainNet(nn.Module):
    def __init__(self,
                 backbone,
                 transformer,
                 criterion,
                 num_classes=dict(
                     entity_labels=0,
                     rel_labels=0
                 ),
                 entity_queries=25,
                 rel_num_queries=30,
                 id_emb_dim=8,
                 aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: dict of number of sub clses, obj clses and relation clses,
                         omitting the special no-object category
                         keys: ["obj_labels", "rel_labels"]
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.entity_queries = entity_queries
        self.rel_num_queries = rel_num_queries
        self.backbone = backbone
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # instance branch
        self.class_embed = nn.Linear(hidden_dim, num_classes['entity_labels'] + 1)
        # 转化为分类问题
        self.pos_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(entity_queries, hidden_dim)
        self.criterion = criterion

    def forward(self, data):
        input_ids = data['input_ids']
        mask = data['mask']

        # backbone
        last_hidden = self.backbone(input_ids, mask)

        # encoder + decoders
        hs = self.transformer(last_hidden, mask, self.query_embed.weight)

        # FFN on top of the instance decoder
        outputs_class = self.class_embed(hs)
        outputs_pos = self.pos_embed(hs)

        out = {'pred_logits': outputs_class, 'pred_pos': outputs_pos}
        output = {
            'pred_entity': out,
            'pred_rel': None,
        }
        loss_dict = self.criterion(output, data)

        return loss_dict

    def evaluate(self, data):
        input_ids = data['input_ids']
        mask = data['mask']

        # backbone
        last_hidden = self.backbone(input_ids, mask)

        # encoder + decoders
        hs = self.transformer(last_hidden, mask, self.query_embed.weight)

        # FFN on top of the instance decoder
        outputs_class = self.class_embed(hs)
        outputs_pos = self.pos_embed(hs)

        out = {'pred_logits': outputs_class, 'pred_pos': outputs_pos}
        output = {
            'pred_entity': out,
            'pred_rel': None,
        }
        t_label, o_label = self.criterion.evaluation_with_match(output, data)

        return t_label, o_label
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterion(nn.Module):
    """ This class computes the loss for HOI Transformer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self,
                 matcher,
                 losses,
                 weight_dict,
                 eos_coef,
                 num_classes=dict(
                     obj_labels=1,
                     rel_labels=1
                 ),
                 neg_act_id=0):
        """ Create the criterion.
        Parameters:
            num_classes: dict of number of sub clses, obj clses and relation clses,
                         omitting the special no-object category
                         keys: ["obj_labels", "rel_labels"]
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes['entity_labels']
        self.rel_classes = num_classes['rel_labels']
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs_dict, targets, indices_dict, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_entity' in outputs_dict
        outputs = outputs_dict['pred_entity']
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        # indices 表示的是预测和目标的对应关系
        indices = indices_dict['entity']
        # idx (对应的bs, 对应的quire_id)
        idx = self._get_src_permutation_idx(indices)
        tgt_ids = [torch.tensor([span[2].numpy().tolist() for span in v]) for v in targets]
        # 借助 col_inx 找到正确的排列顺序
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(tgt_ids, indices)]).cuda()
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device).transpose(1, 0)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.permute(1, 0, 2).flatten(0, 1), target_classes.flatten(0, 1),
                                  self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits.transpose(1, 0)[idx], target_classes_o)[0]
        return losses

    def loss_actions(self, outputs_dict, targets, indices_dict, num_boxes_dict, log=True,
                     neg_act_id=0, topk=5, alpha=0.25, gamma=2, loss_reduce='sum'):
        """Intereaction classificatioon loss (multi-label Focal Loss based on Sigmoid)
        targets dicts must contain the key "actions" containing a tensor of dim [nb_target_boxes]
        Return:
            losses keys:["rel_loss_ce", "rel_class_error"]
        """
        assert 'pred_rel' in outputs_dict
        outputs = outputs_dict['pred_rel']
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        indices = indices_dict['rel']
        idx = self._get_src_permutation_idx(indices)

        target_classes_obj = torch.cat(
            [t["rel_labels"][J].to(src_logits.device) for t, (_, J) in zip(targets, indices)])

        target_classes = torch.zeros(src_logits.shape[0], src_logits.shape[1],
                                     self.rel_classes).type_as(src_logits).to(src_logits.device)
        target_classes[idx] = target_classes_obj.type_as(src_logits)
        losses = {}
        pred_sigmoid = src_logits.sigmoid()
        label = target_classes.long()
        pt = (1 - pred_sigmoid) * label + pred_sigmoid * (1 - label)
        focal_weight = (alpha * label + (1 - alpha) * (1 - label)) * pt.pow(gamma)
        rel_loss = F.binary_cross_entropy_with_logits(src_logits,
                                                      target_classes, reduction='none') * focal_weight
        if loss_reduce == 'mean':
            losses['rel_loss_ce'] = rel_loss.mean()
        else:
            losses['rel_loss_ce'] = rel_loss.sum()
        if log:
            _, pred = src_logits[idx].topk(topk, 1, True, True)
            acc = 0.0
            for tid, target in enumerate(target_classes_obj):
                tgt_idx = torch.where(target == 1)[0]
                if len(tgt_idx) == 0:
                    continue
                acc_pred = 0.0
                for tgt_rel in tgt_idx:
                    acc_pred += (tgt_rel in pred[tid])
                acc += acc_pred / len(tgt_idx)
            rel_labels_error = 100 - 100 * acc / len(target_classes_obj)
            losses['rel_class_error'] = torch.from_numpy(np.array(
                rel_labels_error)).to(src_logits.device).float()
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs_dict, targets):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        计算我们找到实体的数量是否正确
        """
        assert 'pred_entity' in outputs_dict
        outputs = outputs_dict['pred_entity']
        assert 'pred_logits' in outputs
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    @torch.no_grad()
    def loss_rel_cardinality(self, outputs_dict, targets, indices_dict, neg_act_id=0):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        assert 'pred_rel' in outputs_dict
        outputs = outputs_dict['pred_rel']
        assert 'pred_logits' in outputs
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["rel_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != neg_act_id).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'rel_cardinality_error': card_err}
        return losses

    def loss_pos(self, outputs_dict, targets, indices_dict):
        """Compute the losses related to the bounding pos, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_entity' in outputs_dict
        outputs = outputs_dict['pred_entity']
        assert 'pred_pos' in outputs

        indices = indices_dict['entity']
        num_span = torch.as_tensor([len(v) for v in targets]).sum()
        idx = self._get_src_permutation_idx(indices)
        tgt_span = [torch.tensor([span[:2].numpy().tolist() for span in v]) for v in targets]
        src_pos = outputs['pred_pos'].transpose(1, 0)[idx]
        # 借助 col_inx 找到正确的排列顺序
        target_pos = torch.cat([t[i] for t, (_, i) in zip(tgt_span, indices)], dim=0).cuda()

        loss_pos = F.l1_loss(src_pos, target_pos, reduction='none')

        losses = {}
        losses['loss_pos'] = loss_pos.sum() / num_span

        return losses

    def loss_rel_vecs(self, outputs_dict, targets, indices_dict, num_boxes_dict):
        """Compute the losses related to the interaction vector, the L1 regression loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_rel' in outputs_dict
        outputs = outputs_dict['pred_rel']
        assert 'pred_boxes' in outputs
        indices = indices_dict['rel']
        num_vecs = num_boxes_dict['rel']
        idx = self._get_src_permutation_idx(indices)
        self.out_idx = idx
        self.tgt_idx = self._get_tgt_permutation_idx(indices)
        src_vecs = outputs['pred_boxes'][idx]
        target_vecs = torch.cat([t['rel_vecs'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_vecs, target_vecs, reduction='none')
        losses = {}
        losses['rel_loss_bbox'] = loss_bbox.sum() / num_vecs
        return losses

    def loss_emb_push(self, outputs_dict, targets, indices_dict, num_boxes_dict, margin=8):
        """id embedding push loss.
        """
        indices = indices_dict['det']
        idx = self._get_src_permutation_idx(indices)
        if len(idx) == 0:
            losses = {'loss_push': torch.Tensor([0.]).mean().to(idx.device)}
            return losses
        id_emb = outputs_dict['pred_det']['id_emb'][idx]
        n = id_emb.shape[0]
        m = [m.reshape(-1) for m in torch.meshgrid(torch.arange(n), torch.arange(n))]
        mask = torch.where(m[1] < m[0])[0]
        emb_cmp = id_emb[m[0][mask]] - id_emb[m[1][mask]]
        emb_dist = torch.pow(torch.sum(torch.pow(emb_cmp, 2), 1), 0.5)
        loss_push = torch.pow((margin - emb_dist).clamp(0), 2).mean()
        losses = {'loss_push': loss_push}
        return losses

    def loss_emb_pull(self, outputs_dict, targets, indices_dict, num_boxes_dict):
        """id embedding pull loss.
        """
        det_indices = indices_dict['det']
        rel_indices = indices_dict['rel']

        # get indices: det_idx1: [rel_idx1_src, rel_idx2_dst]
        det_pred_idx = self._get_src_permutation_idx(det_indices)
        target_det_centr = torch.cat([t['boxes'][i] for t, (_, i) in zip(
            targets, det_indices)], dim=0)[..., :2]
        rel_pred_idx = self._get_src_permutation_idx(rel_indices)
        if len(rel_pred_idx) == 0:
            losses = {'loss_pull': torch.Tensor([0.]).mean().to(rel_pred_idx.device)}
            return losses
        target_rel_centr = torch.cat([t['rel_vecs'][i] for t, (_, i) in zip(
            targets, rel_indices)], dim=0)
        src_emb = outputs_dict['pred_rel']['src_emb'][rel_pred_idx]
        dst_emb = outputs_dict['pred_rel']['dst_emb'][rel_pred_idx]
        id_emb = outputs_dict['pred_det']['id_emb'][det_pred_idx]

        ref_id_emb = []
        for i in range(len(src_emb)):
            ref_idx = torch.where(target_det_centr == target_rel_centr[i, :2])[0]
            if len(ref_idx) == 0:
                # to remove cur instead of setting to 0.
                losses = {'loss_pull': torch.Tensor([0.]).mean().to(ref_idx.device)}
                return losses
            ref_id_emb.append(id_emb[ref_idx[0]])
        for i in range(len(dst_emb)):
            ref_idx = torch.where(target_det_centr == target_rel_centr[i, 2:])[0]
            if len(ref_idx) == 0:
                losses = {'loss_pull': torch.Tensor([0.]).mean().to(ref_idx.device)}
                return losses
            ref_id_emb.append(id_emb[ref_idx[0]])
        pred_rel_emb = torch.cat([src_emb, dst_emb], 0)
        ref_id_emb = torch.stack(ref_id_emb, 0).to(pred_rel_emb.device)
        loss_pull = torch.pow((pred_rel_emb - ref_id_emb), 2).mean()
        losses = {'loss_pull': loss_pull}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_neg_permutation_idx(self, neg_indices):
        # permute neg rel predictions following indices
        batch_idx = torch.cat([torch.full_like(neg_ind, i) for i, neg_ind in enumerate(neg_indices)])
        neg_idx = torch.cat([neg_ind for neg_ind in neg_indices])
        return batch_idx, neg_idx

    def _targets_transforme(self, targets):
        targets = targets['entities']
        targets = targets.cpu().numpy().tolist()
        for i, v in enumerate(targets):
            del_id = len(v)
            for j, span in enumerate(v):
                if span == ENTITY_PADDING:
                    del_id = j
                    break
            targets[i] = torch.tensor(targets[i][:del_id])
        return targets

    def get_loss(self, loss, outputs_dict, targets, indices_dict, **kwargs):
        targets = self._targets_transforme(targets)
        if outputs_dict['pred_rel'] is None:
            loss_map = {
                'labels': self.loss_labels,
                'cardinality': self.loss_cardinality,
                'pos': self.loss_pos
            }
        else:
            loss_map = {
                'labels': self.loss_labels,
                'cardinality': self.loss_cardinality,
                'pos': self.loss_pos,
                'actions': self.loss_actions,
                'rel_vecs': self.loss_rel_vecs,
                'rel_cardinality': self.loss_rel_cardinality,
                'emb_push': self.loss_emb_push,
                'emb_pull': self.loss_emb_pull
            }
        if loss not in loss_map:
            return {}
        return loss_map[loss](outputs_dict, targets, indices_dict, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        indices_dict = self.matcher(outputs, targets)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets,
                                        indices_dict))
        # loss_tot = losses['loss_ce'] + losses['loss_pos']
        return losses

    def evaluation_with_match(self, outputs, targets):
        # 首先通过二分的方法找到preds和targets最佳的对应方式
        # 对于preds的位置去整，判断位置是否正确，正确的位置将预测的类别加入序列 错误的位置加入一个特殊的类标标记该实体位置不对
        # 将预测序列使用 classification_report库计算f1

        # tuple (row=predict, col=target) 第row个predict 对应第 col个target
        indices_dict = self.matcher(outputs, targets)['entity']
        # 根据indices选择位置
        query_entity, bs, _ = outputs['pred_entity']['pred_pos'].size()
        o_labels = []
        t_labels = []
        for i in range(bs):
            # [0] 表示取row
            # 对于predict
            index1 = indices_dict[i][0].clone().detach().cuda()
            o_pos = outputs['pred_entity']['pred_pos'].index_select(0, index1)[:, i, :]
            o_max, o_indexes = torch.max(outputs['pred_entity']['pred_logits'], dim=2)
            o_label= o_indexes.index_select(0, index1)[:, i]
            # 对于 target
            index2 = indices_dict[i][1].clone().detach().cuda()
            t = targets['entities'].index_select(1, index2)[i, :, :]
            t_pos = t[:, :2]
            t_label = t[:, 2]
            # 最后需要生成label序列
            # 1. 将o_pos 取整数 2.只有位置正确才会将预测的label放上去
            o_pos_round = o_pos.round().long()
            entity_num, _ = t_pos.size()
            for e in range(entity_num):
                t_labels.append(t_label[e].cpu().numpy().tolist())
                if torch.equal(o_pos_round[e, :], t_pos[e, :]):
                    o_labels.append(o_label[e].cpu().numpy().tolist())
                    continue
                else:
                    o_labels.append(len(task_ner_labels['drug']))
        # label_index = range(len(task_ner_labels['drug'])+1)
        # label_name = task_ner_labels['drug'].append('pos_error')
        # f1 = classification_report(t_labels, o_labels, label_index, label_name)

        return t_labels, o_labels