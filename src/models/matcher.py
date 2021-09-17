import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from data.const import ENTITY_PADDING, RELATION_PADDING


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_pos: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_pos = cost_pos
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs_dict, targets, pos_type='separate'):
        """ Performs the matching

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
            C的意义为 b的q选对第i个答案（i in num_entity in all batch）类别和位置的概率
        """

        # output_pos [bs, q, l]
        # output_cla [q, bs, class]
        outputs = outputs_dict['pred_entity']
        num_queries, bs = outputs["pred_logits"].shape[:2]

        # Also concat the target labels and boxes
        # targets ->[num_queries , bs, 3] [0:start 1:end 2:label]
        # targets -> [bs, num_entity, span=3]
        targets = targets['entities']
        # 这里不得不将所有的数据都移动到cpu中进行操作
        targets = targets.cpu().numpy().tolist()
        for i, v in enumerate(targets):
            # del_id 为最大的长度
            del_id = len(v)
            for j, span in enumerate(v):
                if span == ENTITY_PADDING:
                    del_id = j
                    break
            targets[i] = torch.tensor(targets[i][:del_id])

        # 拼接类别的目标 [b1,...bn]`
        tgt_ids = torch.cat([torch.tensor([span[2].numpy().tolist() for span in v], dtype=torch.long) for v in targets])

        # We flatten to compute the cost matrices in a batch
        # [num_queries, batch_size, num_classes]
        out_prob = outputs["pred_logits"].transpose(1, 0).flatten(0, 1).softmax(
            -1)  # [num_queries_b1 b2 b3 ...b, num_classes]
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # cost_class = [num_query*batch_size=175, nums_entity_in_all batch] -> 这里有选择和复制两个功能
        cost_class = -out_prob[:, tgt_ids]

        if pos_type == 'separate':
            out_pos_start = outputs["pred_pos"][0].flatten(0, 1)  # [batch_size_num_queried_1, n2 ... n25, num_classes]
            out_pos_end = outputs["pred_pos"][1].flatten(0, 1)  # [batch_size * num_queries, num_classes]
            tgt_pos_start = torch.cat(
                [torch.tensor([span[1].numpy().tolist() for span in v], dtype=torch.long) for v in targets]).long()
            tgt_pos_end = torch.cat(
                [torch.tensor([span[2].numpy().tolist() for span in v], dtype=torch.long) for v in targets]).long()
            cost_pos = -out_pos_start[:, tgt_pos_start] - out_pos_end[:, tgt_pos_end]
        else:
            # We flatten to compute the cost matrices in a batch
            out_pos = outputs["pred_pos"].flatten(0, 1)  # [batch_size * num_queries, 2]
            # Compute the L1 cost between boxes
            tgt_pos = torch.cat(
                [torch.tensor([span[:2].numpy().tolist() for span in v], dtype=torch.float) for v in targets]).float()
            cost_pos = torch.cdist(out_pos.cpu(), tgt_pos, p=1)

        # Final cost matrix
        C = self.cost_pos * cost_pos.cuda() + self.cost_class * cost_class
        # C = bs, num_queries, sum(target_entity) for all bs
        C = C.view(bs, num_queries, -1).cpu()

        # C.split(sizes, -1) = tuple:4  (bs, num_queries, target_entity)
        # c[i]: per_sentence, num_queries, target_entity
        # row_ind, col_ind = linear_sum_assignment(cost)
        # col_ind表示每一行应该如何选 这里也就是 如何选target_entity
        # 这里利用size把不属于这个batch的内容给去掉
        # 为什么target并没有被分
        sizes = [len(v) for v in targets]
        C = C.split(sizes, -1)
        # 这里再把不同batch的单独摘出来
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C)]
        # list [bs* (row_ind, col_ind)] 每相应的r_ind行选第col_inx列
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        # # for rel
        # rel_outputs = outputs_dict['pred_rel']
        # bs, rel_num_queries = rel_outputs["pred_logits"].shape[:2]
        # rel_out_prob = rel_outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        # rel_out_bbox = rel_outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        # rel_tgt_ids = torch.cat([v["rel_labels"] for v in targets]) # 正确的关系
        # rel_tgt_bbox = torch.cat([v["rel_vecs"] for v in targets])
        #
        # # interaction category semantic distance
        # rel_cost_list = []
        # for idx, r_tgt_id in enumerate(rel_tgt_ids):
        #     tgt_rel_id = torch.where(r_tgt_id == 1)[0]
        #     rel_cost_list.append(-(rel_out_prob[:, tgt_rel_id]).sum(
        #         dim=-1) * self.cost_class)

        indices_dict = {
            'entity': indices
        }

        return indices_dict
