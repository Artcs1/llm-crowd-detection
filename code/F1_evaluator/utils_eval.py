import numpy as np
import json
import os
from graph import graph_propagation
from tqdm import tqdm
import networkx as nx
from community import community_louvain
import markov_clustering as mc

class Data(object):
    def __init__(self, name):
        self.__name = name
        self.__links = set()

    @property
    def name(self):
        return self.__name

    @property
    def links(self):
        return set(self.__links)

    def add_link(self, other, score):
        self.__links.add(other)
        other.__links.add(self)


def connected_components_constraint(nodes, max_sz, score_dict=None, th=None):
    """
    only use edges whose scores are above `th`
    if a component is larger than `max_sz`, all the nodes in this component are added into `remain` and returned for next iteration.
    """
    # print(th)
    result = []
    remain = set()
    nodes = set(nodes)
    while nodes:
        n = nodes.pop()
        group = {n}
        queue = [n]
        valid = True
        while queue:
            n = queue.pop(0)
            if th is not None:
                neighbors = {
                    l
                    for l in n.links
                    if score_dict[tuple(sorted([n.name, l.name]))] >= th
                }
            else:
                neighbors = n.links
            neighbors.difference_update(group)
            nodes.difference_update(neighbors)
            group.update(neighbors)
            queue.extend(neighbors)
            if len(group) > max_sz or len(remain.intersection(neighbors)) > 0:
                # if this group is larger than `max_sz`, add the nodes into `remain`
                # print("Invalid",len(group) > max_sz,len(remain.intersection(neighbors)) > 0)
                valid = False
                remain.update(group)
                break
        if valid:  # if this group is smaller than or equal to `max_sz`, finalize it.
            result.append(group)

    return result, remain


def graph_propagation_kai(edges, score, max_sz, beg_th, step=0.01, pool=None):
    # print(score)
    # score *= 0.99 # to avoid infinite loop
    edges = np.sort(edges, axis=1)
    th = score.min()
    # print("th:",th)
    th = beg_th
    # construct graph[]
    score_dict = {}  # score lookup table
    if pool is None:
        for i, e in enumerate(edges):
            score_dict[e[0], e[1]] = score[i]
    elif pool == "avg":
        for i, e in enumerate(edges):
            if (e[0], e[1]) in score_dict:
                score_dict[e[0], e[1]] = 0.5 * (score_dict[e[0], e[1]] + score[i])
            else:
                score_dict[e[0], e[1]] = score[i]

    elif pool == "max":
        for i, e in enumerate(edges):
            if (e[0], e[1]) in score_dict:
                score_dict[e[0], e[1]] = max(score_dict[e[0], e[1]], score[i])
            else:
                score_dict[e[0], e[1]] = score[i]
    else:
        raise ValueError("Pooling operation not supported")

    nodes = np.sort(np.unique(edges.flatten()))
    mapping = -1 * np.ones((nodes.max() + 1), dtype=int)
    mapping[nodes] = np.arange(nodes.shape[0])
    link_idx = mapping[edges]
    vertex = [Data(n) for n in nodes]
    for l, s in zip(link_idx, score):
        vertex[l[0]].add_link(vertex[l[1]], s)

    # first iteration
    comps, remain = connected_components_constraint(vertex, 0, score_dict, th)

    # iteration
    components = comps[:]
    index = 0
    while remain:
        # print("  ---label prop",index)
        index += 1
        th = th + (1 - th) * step
        comps, remain = connected_components_constraint(remain, max_sz, score_dict, th)
        components.extend(comps)
    return components


class Load_GT:

    def __init__(self, group_path_framewise, group_path, id_path):
        self.group_path = group_path
        self.group_path_framewise = group_path_framewise
        self.id_dict = json.load(open(id_path, "r"))

    def __call__(self, test_list, frame_wise):
        if frame_wise == True:
            scene_wise_group_dict = {}
            for scene_id, test in enumerate(test_list):
                scene_wise_group_dict[scene_id] = {}
                scene_wise_group = []
                data = json.load(
                    open(
                        os.path.join(self.group_path_framewise, test + ".json"),
                        "r",
                        encoding="UTF-8",
                    )
                )
                for frame, info in enumerate(data):
                    scene_wise_group_dict[scene_id][frame] = []
                    for group_info in info.values():
                        group = []
                        for person in group_info["person"]:
                            group.append(person["idx"])
                        scene_wise_group_dict[scene_id][frame].append(group)
            return scene_wise_group_dict
        else:
            scene_wise_group_dict = {}
            for scene_id, test in enumerate(test_list):
                scene_wise_group_dict[scene_id] = []
                data = json.load(
                    open(
                        os.path.join(self.group_path, test + ".json"),
                        "r",
                        encoding="UTF-8",
                    )
                )
                for group_info in data:
                    group = []
                    for person in group_info["person"]:
                        group.append(self.id_dict[str(scene_id)][str(person["idx"])][0])
                    scene_wise_group_dict[scene_id].append(group)
            return scene_wise_group_dict


import torch


class Predict_graph:
    def __init__(self, pre, cfg, id_path,seed):
        self.pre = pre
        self.seed = seed
        self.id_dict = json.load(open(id_path, "r"))
        self.pre = self.preprocess()
        self.cfg = cfg
        self.id_convertion_scene, self.id_inconvertion_scene = (
            self.get_convertion_dict()
        )
        self.pre_LPA = self.convert_id_4_LPA()
        self.edge_score = self.get_score_edge_dic()
        self.LPA_result = self.LPA()
        self.louvain_convertion_id, self.louvain_inconvertion_id = (
            self.get_convertion_dict_louvain()
        )
        self.pre_Louvain = self.convert_id_4_Louvain()

    def preprocess(self):
        new_pre = {}
        for scene_id in self.pre:
            new_pre[scene_id] = {}
            for frame in self.pre[scene_id]:
                new_pre[scene_id][frame] = {}
                for pair, score in self.pre[scene_id][frame].items():
                    pair_1 = self.id_dict[str(scene_id)][str(pair[0])][0]
                    pair_2 = self.id_dict[str(scene_id)][str(pair[1])][0]
                    new_pre[scene_id][frame][(pair_1, pair_2)] = score
        return new_pre

    def get_convertion_dict(self):
        id_convertion_scene = {}
        id_inconvertion_scene = {}
        for scene_id in self.pre:
            id_convertion_scene[scene_id] = {}
            id_inconvertion_scene[scene_id] = {}
            for frame in self.pre[scene_id]:
                id_convertion_scene[scene_id][frame] = {}
                id_inconvertion_scene[scene_id][frame] = {}
                ps_list = []
                for key in self.pre[scene_id][frame]:
                    ps_list.append(key[0])
                    ps_list.append(key[1])
                ps_list = set(sorted(ps_list))
                for i, psid in enumerate(ps_list):
                    id_convertion_scene[scene_id][frame][psid] = i
                    id_inconvertion_scene[scene_id][frame][i] = psid
        return id_convertion_scene, id_inconvertion_scene

    def convert_id_4_LPA(self):
        pre_LPA = {}
        for scene_id in self.pre:
            pre_LPA[scene_id] = {}
            for frame in self.pre[scene_id]:
                pre_LPA[scene_id][frame] = {}
                for key in self.pre[scene_id][frame]:
                    key_convert_0 = self.id_convertion_scene[scene_id][frame][key[0]]
                    key_convert_1 = self.id_convertion_scene[scene_id][frame][key[1]]
                    pre_LPA[scene_id][frame][(key_convert_0, key_convert_1)] = self.pre[
                        scene_id
                    ][frame][key]
        return pre_LPA

    def get_score_edge_dic(self):
        edge_score = {}
        for scene_id in self.pre_LPA:
            edge_score[scene_id] = {}
            for frame in self.pre_LPA[scene_id]:
                edges = [i for i in self.pre_LPA[scene_id][frame].keys()]
                scores = [i[0] for i in self.pre_LPA[scene_id][frame].values()]
                edge_score[scene_id][frame] = {"edges": edges, "scores": scores}
        return edge_score

    def LPA(self):
        edges_score = self.edge_score
        LABEL_PROPAGATION_MAX_SIZE = self.cfg.LABEL_PROPAGATION_MAX_SIZE
        LABEL_PROPAGATION_STEP = self.cfg.LABEL_PROPAGATION_STEP
        LABEL_PROPAGATION_POOL = self.cfg.LABEL_PROPAGATION_POOL
        LP_clustering = {}
        for scene_id in edges_score:
            LP_clustering[scene_id] = {}
            for frame in edges_score[scene_id]:
                edges = edges_score[scene_id][frame]["edges"]
                scores = edges_score[scene_id][frame]["scores"]
                scores = np.array(scores)
                clusters = graph_propagation(
                    edges,
                    scores,
                    max_sz=LABEL_PROPAGATION_MAX_SIZE,
                    step=LABEL_PROPAGATION_STEP,
                    pool=LABEL_PROPAGATION_POOL,
                    beg_th=self.cfg.TH,
                )
                gppsn = 0
                group = []
                for ci, c in enumerate(clusters):
                    group.append([])
                    for xid in c:
                        group[ci].append(xid.name)
                    gppsn += len(group[ci])
                LP_clustering[scene_id][frame] = group
        LPA_clustering_id_restored = {}
        for scene_id in LP_clustering:
            LPA_clustering_id_restored[scene_id] = {}
            for frame, cluster in LP_clustering[scene_id].items():
                group_restore = []
                for group in cluster:
                    res = []
                    for ps in group:
                        res.append(self.id_inconvertion_scene[scene_id][frame][ps])
                    group_restore.append(res)
                LPA_clustering_id_restored[scene_id][frame] = group_restore
        return LPA_clustering_id_restored

    def get_convertion_dict_louvain(self):
        louvain_convertion_id = {}
        louvain_inconvertion_id = {}
        for scene_id in self.LPA_result:
            louvain_convertion_id[scene_id] = {}
            louvain_inconvertion_id[scene_id] = {}
            scene_ps = []
            for frame, cluster in self.LPA_result[scene_id].items():
                for group in cluster:
                    for ps in group:
                        scene_ps.append(ps)
            for i, ps in enumerate(sorted(set(scene_ps))):
                louvain_convertion_id[scene_id][ps] = i
                louvain_inconvertion_id[scene_id][i] = ps
        return louvain_convertion_id, louvain_inconvertion_id

    def convert_id_4_Louvain(self):
        pre_Louvain = {}
        for scene_id in self.pre:
            pre_Louvain[scene_id] = {}
            for frame in self.pre[scene_id]:
                pre_Louvain[scene_id][frame] = {}
                for key in self.pre[scene_id][frame]:
                    key_convert_0 = self.louvain_convertion_id[scene_id][key[0]]
                    key_convert_1 = self.louvain_convertion_id[scene_id][key[1]]
                    pre_Louvain[scene_id][frame][(key_convert_0, key_convert_1)] = (
                        self.pre[scene_id][frame][key]
                    )
        return pre_Louvain

    def listcounter(self, lists):
        dic_list = {}
        dic_counter = {}
        skip = False
        for i, list in enumerate(lists):
            skip = False
            for key, value in dic_list.items():
                if value[0] == list:
                    value[1] += 1
                    skip = True
                    continue
            if skip == False:
                dic_list[i] = [list, 1]
        # print(dic_list)
        max_count = 0
        max_key = -1
        for key, values in dic_list.items():
            if max_count < values[1]:
                max_count = values[1]
                max_key = key
        return dic_list, max_count, max_key

    def louvain_only(self):

        result_dict = {}
        for scene_id in tqdm(self.pre_Louvain):
            add_edges = []
            nodes_set = []
            G = nx.Graph()
            num_in_scene = len(self.louvain_convertion_id[scene_id])
            result_dict[scene_id] = {}
            for frame in self.pre_Louvain[scene_id]:
                for edge, score in self.pre_Louvain[scene_id][frame].items():
                    node = np.array([edge[0], edge[1]]) + num_in_scene * frame
                    nodes_set += node.tolist()
                    if score[0] > self.cfg.SCORE_TH:
                        add_edge_weight = (
                            edge[0] + num_in_scene * frame,
                            edge[1] + num_in_scene * frame,
                            {"weight": (score[0] - self.cfg.TH) / (1 - self.cfg.TH)},
                        )
                        add_edges.append(add_edge_weight)
            for i in range(num_in_scene):
                same_person = sorted(
                    [j for j in set(nodes_set) if j % num_in_scene == i]
                )
                for i in range(len(same_person) - 1):
                    if same_person[i + 1] - same_person[i] == num_in_scene:
                        add_edges.append(
                            (same_person[i], same_person[i + 1], {"weight": 1})
                        )
            G.add_edges_from(add_edges)
            partition_avg = {}
            for i in range(self.cfg.LOUVAIN_LOOP_NUM):
                partition = community_louvain.best_partition(
                    G, resolution=0.5, random_state=self.seed
                )
                grouping = {}
                for i in partition:
                    group = partition[i]
                    if not group in grouping:
                        grouping[group] = [i]
                    else:
                        grouping[group].append(i)
                result = []
                for i in grouping:
                    result.append(grouping[i])
                for frame in range(len(self.pre_Louvain[scene_id])):
                    result_framewise = []
                    for x in result:
                        group = [
                            i - frame * num_in_scene
                            for i in x
                            if i >= frame * num_in_scene
                            and i < (frame + 1) * num_in_scene
                        ]
                        if not len(group) == 0:
                            result_framewise.append(sorted(group))
                    result_dict[scene_id][frame] = result_framewise
                for frame in result_dict[scene_id]:
                    if not frame in partition_avg:
                        partition_avg[frame] = [sorted(result_dict[scene_id][frame])]
                    else:
                        partition_avg[frame].append(
                            sorted(result_dict[scene_id][frame])
                        )
            for frame in partition_avg:
                dic_list, max_count, max_key = self.listcounter(partition_avg[frame])
                result_dict[scene_id][frame] = dic_list[max_key][0]
        clustering_result = {}
        for scene_id in result_dict:
            clustering_result[scene_id] = {}
            for frame, cluster in result_dict[scene_id].items():
                group_converted = []
                for group in cluster:
                    res = []
                    for ps in group:
                        res.append(self.louvain_inconvertion_id[scene_id][ps])
                    group_converted.append(sorted(res))
                clustering_result[scene_id][frame] = group_converted
        self.clustering_result = clustering_result
        return clustering_result

    def clustering_framewise_scene(self):
        result_scene = {}
        for scene_id in range(len(self.clustering_result)):
            result_scene[scene_id] = {}
            res = []
            for frame in self.clustering_result[scene_id]:
                for group in self.clustering_result[scene_id][frame]:
                    if len(group) != 1:
                        res.append(group)
                group_dict = {}
                group_dict2 = {}
                group_count = {}
                for i, group in enumerate(res):
                    if not tuple(group) in group_dict:
                        group_dict[tuple(group)] = i
                        group_dict2[i] = group
                        group_count[i] = 1
                    else:
                        group_count[group_dict[tuple(group)]] += 1
            group_count = sorted(group_count.items(), key=lambda x: x[1], reverse=True)
            ps_list = []
            result = []
            group_big = []
            for key in group_count:
                if key[1] < 0:
                    continue
                group = group_dict2[key[0]]
                group_card = len(group)
                add = 1
                for pre in result:
                    pre_card = len(pre)
                    inters = list(set(pre) & set(group))
                    inters_card = len(inters)
                    if group_card == 2 and pre_card == 2:
                        continue
                    if inters_card / max(pre_card, group_card) >= 0.5:
                        add = 0
                        continue
                if add == 1:
                    result.append(group)

            result_scene[scene_id] = result
        return result_scene

    def clustering_framewise_scene_jrdb(self):
        result_scene = {}
        for scene_id in range(len(self.clustering_result)):
            result_scene[scene_id] = {}
            res = []
            for frame in self.clustering_result[scene_id]:
                for group in self.clustering_result[scene_id][frame]:
                    if len(group) != 1:
                        res.append(group)
                group_dict = {}
                group_dict2 = {}
                group_count = {}
                for i, group in enumerate(res):
                    if not tuple(group) in group_dict:
                        group_dict[tuple(group)] = i
                        group_dict2[i] = group
                        group_count[i] = 1
                    else:
                        group_count[group_dict[tuple(group)]] += 1
            group_count = sorted(group_count.items(), key=lambda x: x[1], reverse=True)
            ps_list = []
            result = []
            for key in group_count:
                group_pre = []
                if key[1] < 0:
                    continue
                group = group_dict2[key[0]]
                group_card = len(group)
                for ps in group:
                    if not ps in ps_list:
                        group_pre.append(ps)
                        ps_list.append(ps)
                if len(group_pre) >= 2:
                    result.append(group_pre)

            result_scene[scene_id] = result
        return result_scene


class Evaluation:

    def __init__(self, whole, alone, pre_dict, GT_dict):
        self.whole = whole
        self.alone = alone
        self.pre_dict = pre_dict
        self.GT_dict = GT_dict

    def add_alone(self, whole, gt):
        for scene_id in gt:
            gt_ps_list = []
            for group in gt[scene_id]:
                for ps in group:
                    gt_ps_list.append(ps)
            pre_ps_list = []
            for group in whole[scene_id]:
                for ps in group:
                    pre_ps_list.append(ps)
            for ps in gt_ps_list:
                if not ps in pre_ps_list:
                    whole[scene_id].append([ps])
        return whole

    def group_eval(self, pre, GT, crit="half"):
        if not GT:
            print("GT is empty")
            return

        TP = 0
        num_GT = 0
        num_GROUP = 0
        #print("Evaluating Grouping Performance...")
        if self.whole == False:
            for frame in GT:
                GT_frame = GT[frame]
                if frame not in pre:
                    GROUP_frame = []
                else:
                    GROUP_frame = pre[frame]
                if self.alone == False:
                    GT_frame = [i for i in GT_frame if len(i) != 1]
                    GROUP_frame = [i for i in GROUP_frame if len(i) != 1]
                num_GT += len(GT_frame)
                num_GROUP += len(GROUP_frame)
                for gt in GT_frame:
                    gt_set = set(gt)
                    gt_card = len(gt)
                    for group in GROUP_frame:
                        group_set = set(group)
                        group_card = len(group)
                        if crit == "half":
                            inters = list(gt_set & group_set)
                            inters_card = len(inters)
                            if group_card == 2 and gt_card == 2:
                                if not len(gt_set - group_set):
                                    TP += 1
                            elif inters_card / max(gt_card, group_card) > 1 / 2:
                                TP += 1
                        elif crit == "card":
                            inters = list(gt_set & group_set)
                            inters_card = len(inters)
                            if group_card == 2 and gt_card == 2:
                                if not len(gt_set - group_set):
                                    TP += 1
                            elif inters_card / max(gt_card, group_card) > 2 / 3:
                                TP += 1
                        elif crit == "dpmm":
                            inters = list(gt_set & group_set)
                            inters_card = len(inters)
                            if group_card == 2 and gt_card == 2:
                                if not len(gt_set - group_set):
                                    TP += 1
                            elif inters_card / max(gt_card, group_card) > 0.6:
                                TP += 1
                        elif crit == "all":
                            if not len(gt_set - group_set):
                                TP += 1

        else:
            if self.alone == False:
                GT = [i for i in GT if len(i) != 1]
                pre = [j for j in pre if len(j) != 1]
            num_GT += len(GT)
            num_GROUP += len(pre)
            for gt in GT:
                gt_set = set(gt)
                gt_card = len(gt)
                for group in pre:
                    group_set = set(group)
                    group_card = len(group)
                    if crit == "half":
                        inters = list(gt_set & group_set)
                        inters_card = len(inters)
                        if group_card == 2 and gt_card == 2:
                            if not len(gt_set - group_set):
                                TP += 1
                        elif inters_card / max(gt_card, group_card) > 1 / 2:
                            TP += 1
                    elif crit == "card":
                        inters = list(gt_set & group_set)
                        inters_card = len(inters)
                        if group_card == 2 and gt_card == 2:
                            if not len(gt_set - group_set):
                                TP += 1
                        elif inters_card / max(gt_card, group_card) > 2 / 3:
                            TP += 1
                    elif crit == "dpmm":
                        inters = list(gt_set & group_set)
                        inters_card = len(inters)
                        if group_card == 2 and gt_card == 2:
                            if not len(gt_set - group_set):
                                TP += 1
                        elif inters_card / max(gt_card, group_card) > 0.6:
                            TP += 1
                    elif crit == "all":
                        if not len(gt_set - group_set):
                            TP += 1

        FP = num_GROUP - TP
        FN = num_GT - TP

        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)

        if TP + FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)

        f1 = (2 * precision * recall) / (precision + recall + 1e-8)

        return precision, recall, f1, TP, FP, FN

    def __call__(self):
        if self.whole == True and self.alone == True:
            self.pre_dict = self.add_alone(self.pre_dict, self.GT_dict)
        if self.whole == True:
            print("####Static####")
        else:
            print("####dynamic####")
        avg = [0, 0, 0, 0, 0, 0]
        keys = self.GT_dict.keys()
        #print(type(keys))
        #print(keys)
        keys = list(keys)[2::3]
        #print(keys)
        #print(keys)
        #print(len(keys))
        #for i in range(len(self.GT_dict)):
        for i in keys:
            if i in self.pre_dict:
                result = self.group_eval(self.pre_dict[i], self.GT_dict[i])
            else:
                result = self.group_eval([], self.GT_dict[i])
            if self.whole == True:
                #print("PRE : ", sorted(self.pre_dict[i]))
                #print("GT : ", sorted(self.GT_dict[i]))
                pass
            #print("---------------------------------------")
            #print(
            #    "scene : {} precison : {:.4f} recall : {:.4f} f1 : {:.4f}".format(
            #        i, result[0], result[1], result[2]
            #    )
            #)
            avg[0] += result[3]
            avg[1] += result[4]
            avg[2] += result[5]
            avg[3] += result[2]
            avg[4] += result[0]
            avg[5] += result[1]
        TP, FP, FN, whole_f1, whole_pc, whole_rc = avg
        #print(TP)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        print("---------------------------------------")
        print(
            "WHOLE precison : {:.4f} recall : {:.4f} f1 : {:.4f}".format(
                precision, recall, f1
            )
        )
        """print(
            "AVERAGE precison : {:.4f} recall : {:.4f} f1 : {:.4f}".format(
                whole_pc / len(self.GT_dict),
                whole_rc / len(self.GT_dict),
                whole_f1 / len(self.GT_dict),
            )
        )"""
        print("---------------------------------------")
        
        return precision, recall, f1

class Load_GT_cafe:

    def __init__(self, root_path):
        self.root_path = root_path
        self.id_dict = json.load(open(os.path.join(root_path, "id_dict_cafe.json"), "r"))

    def __call__(self, test_list, frame_wise):
        group_dict = {}
        if frame_wise == True:
            seq_list = test_list
            for seq in seq_list:
                if seq ==  "gt_tracks.pkl" or seq ==  "id_dict_cafe.json":
                        continue
                seq_path = os.path.join(self.root_path,seq)
                clip_list = os.listdir(seq_path)
                clip_list = sorted([int(i) for i in clip_list])
            
                for clip in clip_list:
                    clip_info = json.load(open(os.path.join(seq_path,str(clip),"dynamic.json",), "r"))
                    group_dict[f"{seq}_{clip}"] = dict()
                    for frame, info in clip_info.items():
                        if not int(frame) in group_dict[f"{seq}_{clip}"]:
                            group_dict[f"{seq}_{clip}"][int(frame)] = []
                        for group_info in info.values():
                            group = []
                            for person in group_info["person"]:
                                group.append(person["idx"])
                            group_dict[f"{seq}_{clip}"][int(frame)].append(group)
        else:
            seq_list = test_list
            for seq in seq_list:
                if seq ==  "gt_tracks.pkl" or seq ==  "id_dict_cafe.json":
                        continue
                seq_path = os.path.join(self.root_path,seq)
                clip_list = os.listdir(seq_path)
                clip_list = sorted([int(i) for i in clip_list])
            
                for clip in clip_list:
                    clip_info = json.load(open(os.path.join(seq_path,str(clip),"static.json",), "r"))
                    group_dict[f"{seq}_{clip}"] = []
                    for group_info in clip_info:
                        group = []
                        for person in group_info["person"]:
                            group.append(person["idx"])
                        group_dict[f"{seq}_{clip}"].append(group)
        return group_dict
                
class Evaluation_cafe:

    def __init__(self, whole, alone, pre_dict, GT_dict):
        self.whole = whole
        self.alone = alone
        self.pre_dict = pre_dict
        self.GT_dict = GT_dict

    def add_alone(self, whole, gt):
        for scene_id in gt:
            gt_ps_list = []
            for group in gt[scene_id]:
                for ps in group:
                    gt_ps_list.append(ps)
            pre_ps_list = []
            for group in whole[scene_id]:
                for ps in group:
                    pre_ps_list.append(ps)
            for ps in gt_ps_list:
                if not ps in pre_ps_list:
                    whole[scene_id].append([ps])
        return whole

    def group_eval(self, pre, GT, crit="half"):
        if not GT:
            GT = [[]]
            #return

        TP = 0
        num_GT = 0
        num_GROUP = 0
        #print("Evaluating Grouping Performance...")
        if self.whole == False:
            for frame in GT:
                GT_frame = GT[frame]
                GROUP_frame = pre[frame]
                if self.alone == False:
                    GT_frame = [i for i in GT_frame if len(i) != 1]
                    GROUP_frame = [i for i in GROUP_frame if len(i) != 1]
                num_GT += len(GT_frame)
                num_GROUP += len(GROUP_frame)
                for gt in GT_frame:
                    gt_set = set(gt)
                    gt_card = len(gt)
                    for group in GROUP_frame:
                        group_set = set(group)
                        group_card = len(group)
                        if crit == "half":
                            inters = list(gt_set & group_set)
                            inters_card = len(inters)
                            if group_card == 2 and gt_card == 2:
                                if not len(gt_set - group_set):
                                    TP += 1
                            elif inters_card / max(gt_card, group_card) > 1 / 2:
                                TP += 1
                        elif crit == "card":
                            inters = list(gt_set & group_set)
                            inters_card = len(inters)
                            if group_card == 2 and gt_card == 2:
                                if not len(gt_set - group_set):
                                    TP += 1
                            elif inters_card / max(gt_card, group_card) > 2 / 3:
                                TP += 1
                        elif crit == "dpmm":
                            inters = list(gt_set & group_set)
                            inters_card = len(inters)
                            if group_card == 2 and gt_card == 2:
                                if not len(gt_set - group_set):
                                    TP += 1
                            elif inters_card / max(gt_card, group_card) > 0.6:
                                TP += 1
                        elif crit == "all":
                            if not len(gt_set - group_set):
                                TP += 1

        else:
            if self.alone == False:
                GT = [i for i in GT if len(i) != 1]
                pre = [j for j in pre if len(j) != 1]
            num_GT += len(GT)
            num_GROUP += len(pre)
            for gt in GT:
                gt_set = set(gt)
                gt_card = len(gt)
                for group in pre:
                    group_set = set(group)
                    group_card = len(group)
                    if crit == "half":
                        inters = list(gt_set & group_set)
                        inters_card = len(inters)
                        if group_card == 2 and gt_card == 2:
                            if not len(gt_set - group_set):
                                TP += 1
                        elif inters_card / max(gt_card, group_card) > 1 / 2:
                            TP += 1
                    elif crit == "card":
                        inters = list(gt_set & group_set)
                        inters_card = len(inters)
                        if group_card == 2 and gt_card == 2:
                            if not len(gt_set - group_set):
                                TP += 1
                        elif inters_card / max(gt_card, group_card) > 2 / 3:
                            TP += 1
                    elif crit == "dpmm":
                        inters = list(gt_set & group_set)
                        inters_card = len(inters)
                        if group_card == 2 and gt_card == 2:
                            if not len(gt_set - group_set):
                                TP += 1
                        elif inters_card / max(gt_card, group_card) > 0.6:
                            TP += 1
                    elif crit == "all":
                        if not len(gt_set - group_set):
                            TP += 1

        FP = num_GROUP - TP
        FN = num_GT - TP

        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)

        if TP + FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)

        f1 = (2 * precision * recall) / (precision + recall + 1e-8)

        return precision, recall, f1, TP, FP, FN

    def __call__(self):
        if self.whole == True and self.alone == True:
            self.pre_dict = self.add_alone(self.pre_dict, self.GT_dict)
        if self.whole == True:
            print("####Static####")
        else:
            print("####dynamic####")
        avg = [0, 0, 0, 0, 0, 0]
        for i in self.pre_dict:
            result = self.group_eval(self.pre_dict[i], self.GT_dict[i])
            if self.whole == True:
                #print("PRE : ", sorted(self.pre_dict[i]))
                #print("GT : ", sorted(self.GT_dict[i]))
                pass
            #print("---------------------------------------")
            #print(
            #    "scene : {} precison : {:.4f} recall : {:.4f} f1 : {:.4f}".format(
            #        i, result[0], result[1], result[2]
            #    )
            #)
            avg[0] += result[3]
            avg[1] += result[4]
            avg[2] += result[5]
            avg[3] += result[2]
            avg[4] += result[0]
            avg[5] += result[1]
        TP, FP, FN, whole_f1, whole_pc, whole_rc = avg
        #print(TP)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        print("---------------------------------------")
        print(
            "WHOLE precison : {:.3f} recall : {:.3f} f1 : {:.3f}".format(
                precision, recall, f1
            )
        )
        """print(
            "AVERAGE precison : {:.4f} recall : {:.4f} f1 : {:.4f}".format(
                whole_pc / len(self.GT_dict),
                whole_rc / len(self.GT_dict),
                whole_f1 / len(self.GT_dict),
            )
        )"""
        print("---------------------------------------")
        
        return precision, recall, f1

class Predict_graph_cafe:
    def __init__(self, pre, cfg, id_path,seed):
        self.pre = pre
        self.seed = seed
        self.id_dict = json.load(open(id_path, "r"))
        self.pre = self.preprocess()
        self.cfg = cfg
        self.id_convertion_scene, self.id_inconvertion_scene = (
            self.get_convertion_dict()
        )
        self.pre_LPA = self.convert_id_4_LPA()
        self.edge_score = self.get_score_edge_dic()
        self.LPA_result = self.LPA()
        self.louvain_convertion_id, self.louvain_inconvertion_id = (
            self.get_convertion_dict_louvain()
        )
        self.pre_Louvain = self.convert_id_4_Louvain()

    def preprocess(self):
        new_pre = {}
        for scene_id in self.pre:
            new_pre[scene_id] = {}
            for frame in self.pre[scene_id]:
                new_pre[scene_id][frame] = {}
                for pair, score in self.pre[scene_id][frame].items():
                    pair_1 = self.id_dict[str(scene_id)][str(pair[0])][0]
                    pair_2 = self.id_dict[str(scene_id)][str(pair[1])][0]
                    new_pre[scene_id][frame][(pair_1, pair_2)] = score
        return new_pre

    def get_convertion_dict(self):
        id_convertion_scene = {}
        id_inconvertion_scene = {}
        for scene_id in self.pre:
            id_convertion_scene[scene_id] = {}
            id_inconvertion_scene[scene_id] = {}
            for frame in self.pre[scene_id]:
                id_convertion_scene[scene_id][frame] = {}
                id_inconvertion_scene[scene_id][frame] = {}
                ps_list = []
                for key in self.pre[scene_id][frame]:
                    ps_list.append(key[0])
                    ps_list.append(key[1])
                ps_list = set(sorted(ps_list))
                for i, psid in enumerate(ps_list):
                    id_convertion_scene[scene_id][frame][psid] = i
                    id_inconvertion_scene[scene_id][frame][i] = psid
        return id_convertion_scene, id_inconvertion_scene

    def convert_id_4_LPA(self):
        pre_LPA = {}
        for scene_id in self.pre:
            pre_LPA[scene_id] = {}
            for frame in self.pre[scene_id]:
                pre_LPA[scene_id][frame] = {}
                for key in self.pre[scene_id][frame]:
                    key_convert_0 = self.id_convertion_scene[scene_id][frame][key[0]]
                    key_convert_1 = self.id_convertion_scene[scene_id][frame][key[1]]
                    pre_LPA[scene_id][frame][(key_convert_0, key_convert_1)] = self.pre[
                        scene_id
                    ][frame][key]
        return pre_LPA

    def get_score_edge_dic(self):
        edge_score = {}
        for scene_id in self.pre_LPA:
            edge_score[scene_id] = {}
            for frame in self.pre_LPA[scene_id]:
                edges = [i for i in self.pre_LPA[scene_id][frame].keys()]
                scores = [i[0] for i in self.pre_LPA[scene_id][frame].values()]
                edge_score[scene_id][frame] = {"edges": edges, "scores": scores}
        return edge_score

    def LPA(self):
        edges_score = self.edge_score
        LABEL_PROPAGATION_MAX_SIZE = self.cfg.LABEL_PROPAGATION_MAX_SIZE
        LABEL_PROPAGATION_STEP = self.cfg.LABEL_PROPAGATION_STEP
        LABEL_PROPAGATION_POOL = self.cfg.LABEL_PROPAGATION_POOL
        LP_clustering = {}
        for scene_id in edges_score:
            LP_clustering[scene_id] = {}
            for frame in edges_score[scene_id]:
                edges = edges_score[scene_id][frame]["edges"]
                if edges == []:
                    LP_clustering[scene_id][frame] = []
                    continue
                scores = edges_score[scene_id][frame]["scores"]
                scores = np.array(scores)
                clusters = graph_propagation(
                    edges,
                    scores,
                    max_sz=LABEL_PROPAGATION_MAX_SIZE,
                    step=LABEL_PROPAGATION_STEP,
                    pool=LABEL_PROPAGATION_POOL,
                    beg_th=self.cfg.TH,
                )
                gppsn = 0
                group = []
                for ci, c in enumerate(clusters):
                    group.append([])
                    for xid in c:
                        group[ci].append(xid.name)
                    gppsn += len(group[ci])
                LP_clustering[scene_id][frame] = group
        LPA_clustering_id_restored = {}
        for scene_id in LP_clustering:
            LPA_clustering_id_restored[scene_id] = {}
            for frame, cluster in LP_clustering[scene_id].items():
                group_restore = []
                for group in cluster:
                    res = []
                    for ps in group:
                        res.append(self.id_inconvertion_scene[scene_id][frame][ps])
                    group_restore.append(res)
                LPA_clustering_id_restored[scene_id][frame] = group_restore
        return LPA_clustering_id_restored

    def get_convertion_dict_louvain(self):
        louvain_convertion_id = {}
        louvain_inconvertion_id = {}
        for scene_id in self.LPA_result:
            louvain_convertion_id[scene_id] = {}
            louvain_inconvertion_id[scene_id] = {}
            scene_ps = []
            for frame, cluster in self.LPA_result[scene_id].items():
                for group in cluster:
                    for ps in group:
                        scene_ps.append(ps)
            for i, ps in enumerate(sorted(set(scene_ps))):
                louvain_convertion_id[scene_id][ps] = i
                louvain_inconvertion_id[scene_id][i] = ps
        return louvain_convertion_id, louvain_inconvertion_id

    def convert_id_4_Louvain(self):
        pre_Louvain = {}
        for scene_id in self.pre:
            pre_Louvain[scene_id] = {}
            for frame in self.pre[scene_id]:
                pre_Louvain[scene_id][frame] = {}
                for key in self.pre[scene_id][frame]:
                    key_convert_0 = self.louvain_convertion_id[scene_id][key[0]]
                    key_convert_1 = self.louvain_convertion_id[scene_id][key[1]]
                    pre_Louvain[scene_id][frame][(key_convert_0, key_convert_1)] = (
                        self.pre[scene_id][frame][key]
                    )
        return pre_Louvain

    def listcounter(self, lists):
        dic_list = {}
        dic_counter = {}
        skip = False
        for i, list in enumerate(lists):
            skip = False
            for key, value in dic_list.items():
                if value[0] == list:
                    value[1] += 1
                    skip = True
                    continue
            if skip == False:
                dic_list[i] = [list, 1]
        # print(dic_list)
        max_count = 0
        max_key = -1
        for key, values in dic_list.items():
            if max_count < values[1]:
                max_count = values[1]
                max_key = key
        return dic_list, max_count, max_key

    def louvain_only(self):

        result_dict = {}
        for scene_id in tqdm(self.pre_Louvain):
            add_edges = []
            nodes_set = []
            G = nx.Graph()
            num_in_scene = len(self.louvain_convertion_id[scene_id])
            result_dict[scene_id] = {}
            for frame in self.pre_Louvain[scene_id]:
                for edge, score in self.pre_Louvain[scene_id][frame].items():
                    node = np.array([edge[0], edge[1]]) + num_in_scene * frame
                    nodes_set += node.tolist()
                    if score[0] > self.cfg.SCORE_TH:
                        add_edge_weight = (
                            edge[0] + num_in_scene * frame,
                            edge[1] + num_in_scene * frame,
                            {"weight": (score[0] - self.cfg.TH) / (1 - self.cfg.TH)},
                        )
                        add_edges.append(add_edge_weight)
            for i in range(num_in_scene):
                same_person = sorted(
                    [j for j in set(nodes_set) if j % num_in_scene == i]
                )
                for i in range(len(same_person) - 1):
                    if same_person[i + 1] - same_person[i] == num_in_scene:
                        add_edges.append(
                            (same_person[i], same_person[i + 1], {"weight": 1})
                        )
            G.add_edges_from(add_edges)
            partition_avg = {}
            for i in range(self.cfg.LOUVAIN_LOOP_NUM):
                partition = community_louvain.best_partition(
                    G, resolution=0.5, random_state=self.seed
                )
                grouping = {}
                for i in partition:
                    group = partition[i]
                    if not group in grouping:
                        grouping[group] = [i]
                    else:
                        grouping[group].append(i)
                result = []
                for i in grouping:
                    result.append(grouping[i])
                for frame in self.pre_Louvain[scene_id]:
                    result_framewise = []
                    for x in result:
                        group = [
                            i - frame * num_in_scene
                            for i in x
                            if i >= frame * num_in_scene
                            and i < (frame + 1) * num_in_scene
                        ]
                        if not len(group) == 0:
                            result_framewise.append(sorted(group))
                    result_dict[scene_id][frame] = result_framewise
                for frame in result_dict[scene_id]:
                    if not frame in partition_avg:
                        partition_avg[frame] = [sorted(result_dict[scene_id][frame])]
                    else:
                        partition_avg[frame].append(
                            sorted(result_dict[scene_id][frame])
                        )
            for frame in partition_avg:
                dic_list, max_count, max_key = self.listcounter(partition_avg[frame])
                result_dict[scene_id][frame] = dic_list[max_key][0]
        clustering_result = {}
        for scene_id in result_dict:
            clustering_result[scene_id] = {}
            for frame, cluster in result_dict[scene_id].items():
                group_converted = []
                for group in cluster:
                    res = []
                    for ps in group:
                        res.append(self.louvain_inconvertion_id[scene_id][ps])
                    group_converted.append(sorted(res))
                clustering_result[scene_id][frame] = group_converted
        self.clustering_result = clustering_result
        return clustering_result

    def clustering(self, mode):

        result_dict = {}
        for scene_id in tqdm(self.pre_Louvain):
            add_edges = []
            nodes_set = []
            G = nx.Graph()
            num_in_scene = len(self.louvain_convertion_id[scene_id])
            result_dict[scene_id] = {}
            for frame in self.pre_Louvain[scene_id]:
                for edge, score in self.pre_Louvain[scene_id][frame].items():
                    node = np.array([edge[0], edge[1]]) + num_in_scene * frame
                    nodes_set += node.tolist()
                    if score[0] > self.cfg.SCORE_TH:
                        add_edge_weight = (
                            edge[0] + num_in_scene * frame,
                            edge[1] + num_in_scene * frame,
                            {"weight": (score[0] - self.cfg.TH) / (1 - self.cfg.TH)},
                        )
                        add_edges.append(add_edge_weight)
            for i in range(num_in_scene):
                same_person = sorted(
                    [j for j in set(nodes_set) if j % num_in_scene == i]
                )
                for i in range(len(same_person) - 1):
                    if same_person[i + 1] - same_person[i] == num_in_scene:
                        add_edges.append(
                            (same_person[i], same_person[i + 1], {"weight": 1})
                        )
            G.add_edges_from(add_edges)
            partition_avg = {}
            
            for i in range(self.cfg.LOUVAIN_LOOP_NUM):
                if mode == "louvain":
                    partition = community_louvain.best_partition(
                        G, resolution=0.5, random_state=self.seed
                    )
                    
                elif mode == "cnm":
                    partition = nx.algorithms.community.greedy_modularity_communities(G)
                    cm = {}
                    res = {}
                    for i,cluster in enumerate(partition):
                        for member in sorted(cluster):
                            cm[member] = i
                    members = sorted(list(cm.keys()))
                    for i in members:
                        res[i] = cm[i]            
                    partition = res
                
                elif mode == "lpa":
                    cm = {}
                    res = {}
                    LABEL_PROPAGATION_MAX_SIZE = len(self.pre_Louvain[scene_id])
                    #LABEL_PROPAGATION_MAX_SIZE = 6
                    LABEL_PROPAGATION_STEP = self.cfg.LABEL_PROPAGATION_STEP
                    LABEL_PROPAGATION_POOL = self.cfg.LABEL_PROPAGATION_POOL
                    LP_clustering = {}
                    edges = []
                    scores = []
                    for edge in G.edges(data=True):
                        edges.append((edge[0],edge[1]))
                        scores.append(edge[2]["weight"])
                    
                    scores = np.array(scores)
                    clusters = graph_propagation(
                        edges,
                        scores,
                        max_sz=LABEL_PROPAGATION_MAX_SIZE,
                        step=LABEL_PROPAGATION_STEP,
                        pool=LABEL_PROPAGATION_POOL,
                        beg_th=self.cfg.SCORE_TH,
                    )
                    gppsn = 0
                    group = []
                    for ci, c in enumerate(clusters):
                        group.append([])
                        for xid in c:
                            group[ci].append(xid.name)
                        gppsn += len(group[ci])
                    for i,cluster in enumerate(group):
                        for member in sorted(cluster):
                            cm[member] = i
                    members = sorted(list(cm.keys()))
                    for i in members:
                        res[i] = cm[i]            
                    partition = res
                    
                elif mode == "mca":
                    cm = {}
                    res = {}
                    matrix = nx.to_scipy_sparse_array(G)
                    result = mc.run_mcl(matrix.todense())
                    clusters = mc.get_clusters(result)
                    for i,cluster in enumerate(clusters):
                        for member in sorted(cluster):
                            cm[member] = i
                    members = sorted(list(cm.keys()))
                    for i in members:
                        res[i] = cm[i]            
                    partition = res

                grouping = {}
                for i in partition:
                    group = partition[i]
                    if not group in grouping:
                        grouping[group] = [i]
                    else:
                        grouping[group].append(i)
                result = []
                for i in grouping:
                    result.append(grouping[i])
                for frame in self.pre_Louvain[scene_id]:
                    result_framewise = []
                    for x in result:
                        group = [
                            i - frame * num_in_scene
                            for i in x
                            if i >= frame * num_in_scene
                            and i < (frame + 1) * num_in_scene
                        ]
                        if not len(group) == 0:
                            result_framewise.append(sorted(group))
                    result_dict[scene_id][frame] = result_framewise
                for frame in result_dict[scene_id]:
                    if not frame in partition_avg:
                        partition_avg[frame] = [sorted(result_dict[scene_id][frame])]
                    else:
                        partition_avg[frame].append(
                            sorted(result_dict[scene_id][frame])
                        )
            for frame in partition_avg:
                dic_list, max_count, max_key = self.listcounter(partition_avg[frame])
                result_dict[scene_id][frame] = dic_list[max_key][0]
            #print(result_dict[scene_id][frame])
        clustering_result = {}
        for scene_id in result_dict:
            clustering_result[scene_id] = {}
            for frame, cluster in result_dict[scene_id].items():
                group_converted = []
                for group in cluster:
                    res = []
                    for ps in group:
                        res.append(self.louvain_inconvertion_id[scene_id][ps])
                    group_converted.append(sorted(res))
                clustering_result[scene_id][frame] = group_converted
        self.clustering_result = clustering_result
        
        return clustering_result
    
    def clustering_framewise_scene(self):
        result_scene = {}
        for scene_id in range(len(self.clustering_result)):
            result_scene[scene_id] = {}
            res = []
            for frame in self.clustering_result[scene_id]:
                for group in self.clustering_result[scene_id][frame]:
                    if len(group) != 1:
                        res.append(group)
                group_dict = {}
                group_dict2 = {}
                group_count = {}
                for i, group in enumerate(res):
                    if not tuple(group) in group_dict:
                        group_dict[tuple(group)] = i
                        group_dict2[i] = group
                        group_count[i] = 1
                    else:
                        group_count[group_dict[tuple(group)]] += 1
            group_count = sorted(group_count.items(), key=lambda x: x[1], reverse=True)
            ps_list = []
            result = []
            group_big = []
            for key in group_count:
                if key[1] < 0:
                    continue
                group = group_dict2[key[0]]
                group_card = len(group)
                add = 1
                for pre in result:
                    pre_card = len(pre)
                    inters = list(set(pre) & set(group))
                    inters_card = len(inters)
                    if group_card == 2 and pre_card == 2:
                        continue
                    if inters_card / max(pre_card, group_card) >= 0.5:
                        add = 0
                        continue
                if add == 1:
                    result.append(group)

            result_scene[scene_id] = result
        return result_scene

    def clustering_framewise_scene_jrdb(self):
        result_scene = {}
        for scene_id in self.clustering_result:
            result_scene[scene_id] = {}
            res = []
            for frame in self.clustering_result[scene_id]:
                for group in self.clustering_result[scene_id][frame]:
                    if len(group) != 1:
                        res.append(group)
                group_dict = {}
                group_dict2 = {}
                group_count = {}
                for i, group in enumerate(res):
                    if not tuple(group) in group_dict:
                        group_dict[tuple(group)] = i
                        group_dict2[i] = group
                        group_count[i] = 1
                    else:
                        group_count[group_dict[tuple(group)]] += 1
            group_count = sorted(group_count.items(), key=lambda x: x[1], reverse=True)
            ps_list = []
            result = []
            for key in group_count:
                group_pre = []
                if key[1] < 0:
                    continue
                group = group_dict2[key[0]]
                group_card = len(group)
                for ps in group:
                    if not ps in ps_list:
                        group_pre.append(ps)
                        ps_list.append(ps)
                if len(group_pre) >= 2:
                    result.append(group_pre)

            result_scene[scene_id] = result
        return result_scene
