# coding: utf-8


import math
import logging
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import tqdm
from scipy.stats import pearsonr
from DEKTNet import DEKTNet
from sklearn.metrics import r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def etl(*args, **kwargs) -> ...:  # pragma: no cover
    """
    extract - transform - load
    """
    pass


def train(*args, **kwargs) -> ...:  # pragma: no cover
    pass


def evaluate(*args, **kwargs) -> ...:  # pragma: no cover
    pass


class KTM(object):
    def __init__(self, *args, **kwargs) -> ...:
        pass

    def train(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def eval(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def save(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def load(self, *args, **kwargs) -> ...:
        raise NotImplementedError

def compute_accuracy_multi_class(all_target, all_pred):
    # # 将模型输出的每个样本的概率最大的类别作为预测类别
    # predicted_labels = np.argmax(all_pred, axis=1)
    # 计算准确率
    accuracy = metrics.accuracy_score(all_target, all_pred)
    return accuracy


def custom_cross_entropy_loss_multi_class(predictions, targets):
    epsilon = 1e-10  # 用于防止对数中的除零错误
    # 将预测概率通过 softmax 转换
    predictions_exp = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    predictions_softmax = predictions_exp / np.sum(predictions_exp, axis=1, keepdims=True)
    # 防止概率为0的情况
    predictions_softmax = np.maximum(epsilon, predictions_softmax)
    predictions_softmax = np.minimum(1 - epsilon, predictions_softmax)
    # 使用 numpy 的广播进行计算
    loss = -np.sum(np.log(predictions_softmax[np.arange(len(targets)), targets]))
    # 对所有样本取平均
    loss /= len(targets)
    return loss


def binary_entropy(target, pred):
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0


def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    y_hat = (all_pred > 0.5).astype(np.float64)
    return metrics.accuracy_score(all_target, y_hat)

def compute_rmse(all_target, all_pred):
    # 计算均方根误差
    return np.sqrt(np.mean((all_target - all_pred)**2))

def compute_mse(all_target, all_pred):
    # 计算均方根误差
    return np.mean((all_target - all_pred)**2)


def compute_brier(all_target, all_pred):
    return np.mean((all_target - all_pred) ** 2)


def compute_ece(all_target, all_pred, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (all_pred >= left) & (all_pred <= right)
        else:
            mask = (all_pred >= left) & (all_pred < right)
        if not np.any(mask):
            continue
        confidence = all_pred[mask].mean()
        accuracy = all_target[mask].mean()
        ece += abs(confidence - accuracy) * (mask.sum() / len(all_target))
    return float(ece)


def collect_response_metrics(all_target, all_pred):
    return {
        "loss": float(binary_entropy(all_target, all_pred)),
        "auc": float(compute_auc(all_target, all_pred)),
        "accuracy": float(compute_accuracy(all_target, all_pred)),
        "rmse": float(compute_rmse(all_target, all_pred)),
        "r2": float(r2_score(all_target, all_pred)),
        "nll": float(binary_entropy(all_target, all_pred)),
        "brier": float(compute_brier(all_target, all_pred)),
        "ece": float(compute_ece(all_target, all_pred)),
    }




def _cpu_generator(seed):
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _rand(shape, device, generator=None):
    return torch.rand(shape, generator=generator).to(device)


def _randn(shape, device, generator=None):
    return torch.randn(shape, generator=generator).to(device)


def _randint(high, shape, device, generator=None):
    if generator is None:
        return torch.randint(high, shape, device=device)
    return torch.randint(high, shape, generator=generator).to(device)


def _sequence_affect_mean(input_e, input_bor, input_conc, input_conf, input_fru):
    affect_values = torch.stack((input_bor, input_conc, input_conf, input_fru), dim=2).float().clamp(0.0, 1.0)
    valid = (input_e > 0).float().unsqueeze(2)
    denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
    baseline = (affect_values * valid).sum(dim=1, keepdim=True) / denom
    return affect_values, baseline.expand_as(affect_values), valid.bool()


def _same_skill_mismatch_replacement(affect_values, baseline, valid, input_s, selected, generator, input_stu=None):
    selected_count = int(selected.sum().detach().cpu().item())
    if input_s is None or selected_count == 0:
        return baseline, {"selected": selected_count, "same_skill": 0, "fallback": selected_count}

    batch_size, seq_len, channels = affect_values.shape
    device_ = affect_values.device
    flat_affect = affect_values.reshape(-1, channels)
    flat_baseline = baseline.reshape(-1, channels)
    flat_valid = valid.squeeze(2).reshape(-1)
    flat_selected = selected.squeeze(2).reshape(-1)
    flat_skill = input_s.reshape(-1)
    flat_row = torch.arange(batch_size, device=device_).unsqueeze(1).expand(batch_size, seq_len).reshape(-1)
    flat_student = input_stu.reshape(-1) if input_stu is not None else flat_row
    selected_indices = torch.nonzero(flat_selected, as_tuple=False).flatten()
    replacement = flat_baseline.clone()

    same_skill_count = 0
    fallback_indices = []
    selected_skills = torch.unique(flat_skill[selected_indices])
    for skill in selected_skills.tolist():
        skill_selected = selected_indices[flat_skill[selected_indices] == int(skill)]
        skill_pool = torch.nonzero(flat_valid & (flat_skill == int(skill)), as_tuple=False).flatten()
        if skill_pool.numel() <= 1:
            fallback_indices.append(skill_selected)
            continue
        pick = skill_pool[_randint(skill_pool.numel(), (skill_selected.numel(),), device_, generator)]
        same_row = flat_student[pick] == flat_student[skill_selected]
        for _ in range(3):
            if not same_row.any():
                break
            reroll = skill_pool[_randint(skill_pool.numel(), (int(same_row.sum().item()),), device_, generator)]
            pick[same_row] = reroll
            same_row = flat_student[pick] == flat_student[skill_selected]
        ok = ~same_row
        if ok.any():
            replacement[skill_selected[ok]] = flat_affect[pick[ok]]
            same_skill_count += int(ok.sum().detach().cpu().item())
        if same_row.any():
            fallback_indices.append(skill_selected[same_row])

    fallback_count = 0
    if fallback_indices:
        fallback_selected = torch.cat(fallback_indices)
        any_pool = torch.nonzero(flat_valid, as_tuple=False).flatten()
        if any_pool.numel() > 1:
            pick = any_pool[_randint(any_pool.numel(), (fallback_selected.numel(),), device_, generator)]
            same_row = flat_student[pick] == flat_student[fallback_selected]
            for _ in range(3):
                if not same_row.any():
                    break
                reroll = any_pool[_randint(any_pool.numel(), (int(same_row.sum().item()),), device_, generator)]
                pick[same_row] = reroll
                same_row = flat_student[pick] == flat_student[fallback_selected]
            ok = ~same_row
            if ok.any():
                replacement[fallback_selected[ok]] = flat_affect[pick[ok]]
        fallback_count = int(fallback_selected.numel())

    info = {"selected": selected_count, "same_skill": int(same_skill_count), "fallback": int(fallback_count)}
    return replacement.reshape_as(affect_values), info


def perturb_affect_inputs(
    input_e,
    input_bor,
    input_conc,
    input_conf,
    input_fru,
    *,
    perturbation="clean",
    rate=0.0,
    noise_std=0.0,
    seed=None,
    input_s=None,
    input_stu=None,
    return_info=False,
):
    if perturbation in {None, "clean", "none"}:
        result = (input_bor, input_conc, input_conf, input_fru)
        if return_info:
            return result, {"perturbation": "clean", "selected": 0, "same_skill": 0, "fallback": 0}
        return result

    generator = _cpu_generator(seed)
    if perturbation == "mixed":
        if generator is None:
            choice = int(torch.randint(0, 3, (1,)).item())
        else:
            choice = int(torch.randint(0, 3, (1,), generator=generator).item())
        perturbation = ("mask", "noise", "mismatch")[choice]

    affect_values, baseline, valid = _sequence_affect_mean(input_e, input_bor, input_conc, input_conf, input_fru)
    rate = float(rate)
    if perturbation == "noise":
        perturbed = (affect_values + _randn(affect_values.shape, affect_values.device, generator) * float(noise_std)).clamp(0.0, 1.0)
        info = {"perturbation": "noise", "selected": int(valid.sum().detach().cpu().item()), "same_skill": 0, "fallback": 0}
    elif perturbation == "mask":
        selected = (_rand(affect_values.shape[:2] + (1,), affect_values.device, generator) < rate) & valid
        perturbed = torch.where(selected, baseline, affect_values)
        info = {"perturbation": "mask", "selected": int(selected.sum().detach().cpu().item()), "same_skill": 0, "fallback": 0}
    elif perturbation == "mismatch":
        selected = (_rand(affect_values.shape[:2] + (1,), affect_values.device, generator) < rate) & valid
        replacement, info = _same_skill_mismatch_replacement(
            affect_values,
            baseline,
            valid,
            input_s,
            selected,
            generator,
            input_stu=input_stu,
        )
        info["perturbation"] = "mismatch"
        perturbed = torch.where(selected, replacement, affect_values)
    else:
        raise ValueError(f"Unsupported perturbation: {perturbation}")

    result = (perturbed[:, :, 0], perturbed[:, :, 1], perturbed[:, :, 2], perturbed[:, :, 3])
    if return_info:
        return result, info
    return result


def _append_record(record_chunks, key, tensor):
    record_chunks.setdefault(key, []).append(tensor.detach().cpu().numpy())


def _finalize_records(record_chunks):
    integer_keys = {"source_problem", "target_problem", "source_skill", "target_skill"}
    records = {}
    for key, value in record_chunks.items():
        if not value:
            continue
        dtype = int if key in integer_keys else float
        records[key] = np.concatenate(value, axis=0).astype(dtype).tolist()
    return records


def train_one_epoch(
    net,
    optimizer,
    criterion,
    criterion_mse,
    batch_size,
    a_data,
    e_data,
    s_data,
    it_data,
    at_data,
    bor_data,
    conc_data,
    conf_data,
    fru_data,
    qd_data,
    sd_data,
    tp_data,
    stu_data,
    _pre_data,
    att_data,
    affect_loss_weight=2.0,
    grad_clip_norm=5.0,
    stability_weight=0.0,
    stability_perturbation="mixed",
    stability_rate=0.4,
    stability_noise_std=0.1,
    stability_seed=None,
    train_perturbation="clean",
    train_perturbation_rate=0.0,
    train_perturbation_noise_std=0.0,
    train_perturbation_seed=None,
    return_summary=False,
):
    net.train()
    n = int(math.ceil(len(e_data) / batch_size))
    shuffled_ind = np.arange(e_data.shape[0])
    np.random.shuffle(shuffled_ind)
    e_data = e_data[shuffled_ind]
    s_data = s_data[shuffled_ind]
    at_data = at_data[shuffled_ind]
    a_data = a_data[shuffled_ind]
    it_data = it_data[shuffled_ind]
    bor_data = bor_data[shuffled_ind]
    conc_data = conc_data[shuffled_ind]
    conf_data = conf_data[shuffled_ind]
    fru_data = fru_data[shuffled_ind]
    # emo_data = emo_data[shuffled_ind]
    sd_data = sd_data[shuffled_ind]
    qd_data = qd_data[shuffled_ind]
    tp_data = tp_data[shuffled_ind]
    stu_data = stu_data[shuffled_ind]
    att_data = att_data[shuffled_ind]
    


    pred_list = []
    target_list = []



    for idx in tqdm.tqdm(range(n), 'Training'):
        optimizer.zero_grad()

        e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size, :]
        s_one_seq = s_data[idx * batch_size: (idx + 1) * batch_size, :]
        at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        it_one_seq = it_data[idx * batch_size: (idx + 1) * batch_size, :]
        bor_one_seq = bor_data[idx * batch_size: (idx + 1) * batch_size, :]
        conc_one_seq = conc_data[idx * batch_size: (idx + 1) * batch_size, :]
        conf_one_seq = conf_data[idx * batch_size: (idx + 1) * batch_size, :]
        fru_one_seq = fru_data[idx * batch_size: (idx + 1) * batch_size, :]
        # emo_one_seq = emo_data[idx * batch_size: (idx + 1) * batch_size, :]
        sd_one_seq = sd_data[idx * batch_size: (idx + 1) * batch_size, :]
        qd_one_seq = qd_data[idx * batch_size: (idx + 1) * batch_size, :]
        tp_one_seq = tp_data[idx * batch_size: (idx + 1) * batch_size, :]
        stu_one_seq = stu_data[idx * batch_size: (idx + 1) * batch_size, :]
        att_one_seq = att_data[idx * batch_size: (idx + 1) * batch_size, :]
        


        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_s = torch.from_numpy(s_one_seq).long().to(device)
        input_at = torch.from_numpy(at_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        target = torch.from_numpy(a_one_seq).float().to(device)
        input_bor = torch.from_numpy(bor_one_seq).float().to(device)
        input_conc = torch.from_numpy(conc_one_seq).float().to(device)
        input_conf = torch.from_numpy(conf_one_seq).float().to(device)
        input_fru = torch.from_numpy(fru_one_seq).float().to(device)
        input_sd = torch.from_numpy(sd_one_seq).long().to(device)
        input_qd = torch.from_numpy(qd_one_seq).long().to(device)
        input_tp = torch.from_numpy(tp_one_seq).long().to(device)
        input_stu = torch.from_numpy(stu_one_seq).long().to(device)
        input_att = torch.from_numpy(att_one_seq).long().to(device) 


        model_input_bor, model_input_conc, model_input_conf, model_input_fru = perturb_affect_inputs(
            input_e,
            input_bor,
            input_conc,
            input_conf,
            input_fru,
            perturbation=train_perturbation,
            rate=train_perturbation_rate,
            noise_std=train_perturbation_noise_std,
            seed=None if train_perturbation_seed is None else int(train_perturbation_seed) + idx,
            input_s=input_s,
            input_stu=input_stu,
        )

        pred , pred_bor, pred_conc, pred_conf, pred_fru  = net(
            input_e,
            input_s,
            input_at,
            target,
            input_it,
            model_input_bor,
            model_input_conc,
            model_input_conf,
            model_input_fru,
            input_qd,
            input_sd,
            input_tp,
            input_att,
        )
        
        mask = input_e[:, 1:] > 0
        masked_pred = pred[:, 1:][mask]
        masked_truth = target[:, 1:][mask]

        mask_pred_bor = pred_bor[:,1:,][mask]
        mask_truth_bor = input_bor[:,1:][mask]

        mask_pred_conc = pred_conc[:,1:,][mask]
        mask_truth_conc = input_conc[:,1:][mask]

        mask_pred_conf = pred_conf[:,1:,][mask]
        mask_truth_conf = input_conf[:,1:][mask]

        mask_pred_fru = pred_fru[:,1:,][mask]
        mask_truth_fru = input_fru[:,1:][mask]


        loss1 = criterion(masked_pred, masked_truth).sum()

        
        loss21 = criterion_mse(mask_pred_bor,mask_truth_bor.float()).sum()  
        loss22 = criterion_mse(mask_pred_conc,mask_truth_conc.float()).sum()    
        loss23 = criterion_mse(mask_pred_conf,mask_truth_conf.float()).sum()         
        loss24 = criterion_mse(mask_pred_fru,mask_truth_fru.float()).sum()   

        loss2 = loss21+loss22+loss23+loss24

        loss = loss1 + affect_loss_weight * loss2
        if stability_weight > 0.0:
            perturbed_bor, perturbed_conc, perturbed_conf, perturbed_fru = perturb_affect_inputs(
                input_e,
                input_bor,
                input_conc,
                input_conf,
                input_fru,
                perturbation=stability_perturbation,
                rate=stability_rate,
                noise_std=stability_noise_std,
                seed=None if stability_seed is None else int(stability_seed) + idx,
                input_s=input_s,
                input_stu=input_stu,
            )
            perturbed_pred, _, _, _, _ = net(
                input_e,
                input_s,
                input_at,
                target,
                input_it,
                perturbed_bor,
                perturbed_conc,
                perturbed_conf,
                perturbed_fru,
                input_qd,
                input_sd,
                input_tp,
                input_att,
            )
            masked_perturbed_pred = perturbed_pred[:, 1:][mask]
            stability_loss = criterion_mse(masked_perturbed_pred, masked_pred.detach()).sum()
            loss = loss + stability_weight * stability_loss
        
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        # y loss
        masked_pred  = masked_pred.detach().cpu().numpy()
        masked_truth = masked_truth.detach().cpu().numpy()
        pred_list.append(masked_pred)  # 多个array
        target_list.append(masked_truth)


    # y
    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    # y
    summary = collect_response_metrics(all_target, all_pred)
    if return_summary:
        return summary
    return summary["loss"], summary["auc"], summary["accuracy"], summary["rmse"], summary["r2"]


def test_one_epoch(
    net,
    batch_size,
    a_data,
    e_data,
    s_data,
    it_data,
    at_data,
    bor_data,
    conc_data,
    conf_data,
    fru_data,
    qd_data,
    sd_data,
    tp_data,
    stu_data,
    _pre_data,
    att_data,
    perturbation="clean",
    perturbation_rate=0.0,
    perturbation_noise_std=0.0,
    perturbation_seed=0,
    return_records=False,
    return_summary=False,
):
    net.eval()
    n = int(math.ceil(len(e_data) / batch_size))

    pred_list = []
    target_list = []
    record_chunks = {}
    perturbation_totals = {"selected": 0.0, "same_skill": 0.0, "fallback": 0.0}

    
    for idx in tqdm.tqdm(range(n), 'Testing'):

        e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size, :]
        s_one_seq = s_data[idx * batch_size: (idx + 1) * batch_size, :]
        at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        it_one_seq = it_data[idx * batch_size: (idx + 1) * batch_size, :]
        bor_one_seq = bor_data[idx * batch_size: (idx + 1) * batch_size, :]
        conc_one_seq = conc_data[idx * batch_size: (idx + 1) * batch_size, :]
        conf_one_seq = conf_data[idx * batch_size: (idx + 1) * batch_size, :]
        fru_one_seq = fru_data[idx * batch_size: (idx + 1) * batch_size, :]
        sd_one_seq = sd_data[idx * batch_size: (idx + 1) * batch_size, :]
        qd_one_seq = qd_data[idx * batch_size: (idx + 1) * batch_size, :]
        tp_one_seq = tp_data[idx * batch_size: (idx + 1) * batch_size, :]
        stu_one_seq = stu_data[idx * batch_size: (idx + 1) * batch_size, :]
        att_one_seq = att_data[idx * batch_size: (idx + 1) * batch_size, :]
  

        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_s = torch.from_numpy(s_one_seq).long().to(device)
        input_at = torch.from_numpy(at_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        target = torch.from_numpy(a_one_seq).float().to(device)
        input_bor = torch.from_numpy(bor_one_seq).float().to(device)
        input_conc = torch.from_numpy(conc_one_seq).float().to(device)
        input_conf = torch.from_numpy(conf_one_seq).float().to(device)
        input_fru = torch.from_numpy(fru_one_seq).float().to(device)
        input_sd = torch.from_numpy(sd_one_seq).long().to(device)
        input_qd = torch.from_numpy(qd_one_seq).long().to(device)
        input_tp = torch.from_numpy(tp_one_seq).long().to(device)
        input_stu = torch.from_numpy(stu_one_seq).long().to(device)
        input_att = torch.from_numpy(att_one_seq).long().to(device)
        (input_bor, input_conc, input_conf, input_fru), perturbation_info = perturb_affect_inputs(
            input_e,
            input_bor,
            input_conc,
            input_conf,
            input_fru,
            perturbation=perturbation,
            rate=perturbation_rate,
            noise_std=perturbation_noise_std,
            seed=None if perturbation_seed is None else int(perturbation_seed) + idx,
            input_s=input_s,
            input_stu=input_stu,
            return_info=True,
        )
        for key in perturbation_totals:
            perturbation_totals[key] += float(perturbation_info[key])


        with torch.no_grad():
            outputs = net(
                input_e,
                input_s,
                input_at,
                target,
                input_it,
                input_bor,
                input_conc,
                input_conf,
                input_fru,
                input_qd,
                input_sd,
                input_tp,
                input_att,
                return_details=return_records,
            )
            if return_records:
                pred, pred_bor, pred_conc, pred_conf, pred_fru, details = outputs
            else:
                pred, pred_bor, pred_conc, pred_conf, pred_fru = outputs
 
            mask = input_e[:, 1:] > 0
            masked_pred = pred[:, 1:][mask].detach().cpu().numpy()
            masked_truth = target[:, 1:][mask].detach().cpu().numpy()

            
            pred_list.append(masked_pred)
            target_list.append(masked_truth)

            if return_records:
                _append_record(record_chunks, "rho", details["rho"][:, :-1][mask])
                _append_record(record_chunks, "prediction", pred[:, 1:][mask])
                _append_record(record_chunks, "target_answer", target[:, 1:][mask])
                _append_record(record_chunks, "source_answer", target[:, :-1][mask])
                _append_record(record_chunks, "source_problem", input_e[:, :-1][mask])
                _append_record(record_chunks, "target_problem", input_e[:, 1:][mask])
                _append_record(record_chunks, "source_skill", input_s[:, :-1][mask])
                _append_record(record_chunks, "target_skill", input_s[:, 1:][mask])
                _append_record(record_chunks, "source_response_time", input_at[:, :-1][mask])
                _append_record(record_chunks, "source_interval_time", input_it[:, :-1][mask])
                _append_record(record_chunks, "bor", input_bor[:, :-1][mask])
                _append_record(record_chunks, "conc", input_conc[:, :-1][mask])
                _append_record(record_chunks, "conf", input_conf[:, :-1][mask])
                _append_record(record_chunks, "fru", input_fru[:, :-1][mask])
                _append_record(record_chunks, "trusted_bor", details["trusted_affect"][:, :-1, 0][mask])
                _append_record(record_chunks, "trusted_conc", details["trusted_affect"][:, :-1, 1][mask])
                _append_record(record_chunks, "trusted_conf", details["trusted_affect"][:, :-1, 2][mask])
                _append_record(record_chunks, "trusted_fru", details["trusted_affect"][:, :-1, 3][mask])
                _append_record(record_chunks, "affect_deviation_mean", details["affect_deviation"][:, :-1].mean(dim=2)[mask])


    # y
    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)


    summary = collect_response_metrics(all_target, all_pred)
    if return_records:
        summary["records"] = _finalize_records(record_chunks)
    perturbation_selected = perturbation_totals["selected"]
    perturbation_same_skill = perturbation_totals["same_skill"]
    perturbation_fallback = perturbation_totals["fallback"]
    summary["perturbation_info"] = {
        "selected": perturbation_selected,
        "same_skill": perturbation_same_skill,
        "fallback": perturbation_fallback,
        "same_skill_rate": 0.0 if perturbation_selected == 0 else perturbation_same_skill / perturbation_selected,
        "fallback_rate": 0.0 if perturbation_selected == 0 else perturbation_fallback / perturbation_selected,
    }
    if return_summary:
        return summary
    return summary["loss"], summary["auc"], summary["accuracy"], summary["rmse"], summary["r2"]

class DEKT(KTM):
    def __init__(
        self,
        n_at,
        n_it,
        n_exercise,
        n_question,
        d_a,
        d_e,
        d_k,
        d_m,
        q_matrix,
        n_qd,
        n_sd,
        n_tp,
        n_att,
        batch_size,
        dropout=0.2,
        eval_batch_size=None,
        graft_mode="off",
        problem_to_skill=None,
        q_gamma=0.0,
        reliability_mode="off",
        init_rho_bias=3.0,
    ):
        super(DEKT, self).__init__()
        q_matrix_tensor = None if q_matrix is None else torch.from_numpy(q_matrix).float().to(device)
        problem_to_skill_tensor = None
        if problem_to_skill is not None:
            problem_to_skill_tensor = torch.from_numpy(problem_to_skill).long().to(device)
        self.dekt_net = DEKTNet(
            n_at,
            n_it,
            n_exercise,
            n_question,
            d_a,
            d_e,
            d_k,
            d_m,
            q_matrix_tensor,
            n_qd=n_qd,
            n_sd=n_sd,
            n_tp=n_tp,
            n_att=n_att,
            dropout=dropout,
            graft_mode=graft_mode,
            problem_to_skill=problem_to_skill_tensor,
            q_gamma=q_gamma,
            reliability_mode=reliability_mode,
            init_rho_bias=init_rho_bias,
        ).to(device)
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size

    def train(
        self,
        train_data,
        test_data=None,
        *,
        epoch: int,
        lr=0.002,
        lr_decay_step=15,
        lr_decay_rate=0.5,
        save_path="params/dekt.params",
        scheduler_name="step",
        min_lr=1e-5,
        early_stop_patience=None,
        affect_loss_weight=2.0,
        grad_clip_norm=5.0,
        stability_weight=0.0,
        stability_perturbation="mixed",
        stability_rate=0.4,
        stability_noise_std=0.1,
        stability_seed=1729,
        train_perturbation="clean",
        train_perturbation_rate=0.0,
        train_perturbation_noise_std=0.0,
        train_perturbation_seed=8191,
        eval_batch_size=None,
        return_summary=False,
    ) -> ...:
        optimizer = torch.optim.Adam(self.dekt_net.parameters(), lr=lr, eps=1e-8, betas=(0.9, 0.999), weight_decay=1e-6)
        if scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_step, gamma=lr_decay_rate)
        elif scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=min_lr)
        else:
            raise ValueError(f"Unsupported scheduler_name: {scheduler_name}")

        criterion = nn.BCELoss(reduction='none')
        criterion_mse = nn.MSELoss(reduction='none')       
        best_train_auc, best_valid_auc = .0, .0
        best_epoch = 0
        history = []
        stale_epochs = 0
        eval_batch_size = eval_batch_size or self.eval_batch_size
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        for idx in range(epoch):
            
            train_summary = train_one_epoch(
                self.dekt_net,
                optimizer,
                criterion,
                criterion_mse,
                self.batch_size,
                *train_data,
                affect_loss_weight=affect_loss_weight,
                grad_clip_norm=grad_clip_norm,
                stability_weight=stability_weight,
                stability_perturbation=stability_perturbation,
                stability_rate=stability_rate,
                stability_noise_std=stability_noise_std,
                stability_seed=None if stability_seed is None else int(stability_seed) + idx * 100000,
                train_perturbation=train_perturbation,
                train_perturbation_rate=train_perturbation_rate,
                train_perturbation_noise_std=train_perturbation_noise_std,
                train_perturbation_seed=None if train_perturbation_seed is None else int(train_perturbation_seed) + idx * 100000,
                return_summary=True,
            )

            print(
                "[Epoch %d] train_loss: %.6f, train_auc: %.6f, train_ece: %.6f"
                % (idx + 1, train_summary["loss"], train_summary["auc"], train_summary["ece"])
            )
            if train_summary["auc"] > best_train_auc:
                best_train_auc = train_summary["auc"]

            if test_data is not None:
                valid_summary = self.eval(test_data, batch_size=eval_batch_size, return_summary=True)
                print(
                    "[Epoch %d] valid_auc: %.6f, valid_accuracy: %.6f, valid_rmse: %.6f, valid_r2: %.6f"
                    % (
                        idx + 1,
                        valid_summary["auc"],
                        valid_summary["accuracy"],
                        valid_summary["rmse"],
                        valid_summary["r2"],
                    )
                )
                history.append(
                    {
                        "epoch": idx + 1,
                        "train": train_summary,
                        "valid": valid_summary,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    }
                )
                if valid_summary["auc"] > best_valid_auc:
                    torch.save(self.dekt_net.state_dict(), save_path)
                    print(f"此时的valida auc:{valid_summary['auc']}")
                    print(f"目前最好的epoch是{idx+1}")
                    best_valid_auc = valid_summary["auc"]
                    best_epoch = idx + 1
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            else:
                history.append(
                    {
                        "epoch": idx + 1,
                        "train": train_summary,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    }
                )

            scheduler.step()

            if early_stop_patience is not None and test_data is not None and stale_epochs >= early_stop_patience:
                print(f"Early stopping triggered at epoch {idx + 1}")
                break

        summary = {
            "best_train_auc": float(best_train_auc),
            "best_valid_auc": float(best_valid_auc),
            "best_epoch": int(best_epoch),
            "history": history,
            "save_path": str(save_path),
            "stability_weight": float(stability_weight),
            "stability_perturbation": stability_perturbation,
            "stability_rate": float(stability_rate),
            "stability_noise_std": float(stability_noise_std),
            "train_perturbation": train_perturbation,
            "train_perturbation_rate": float(train_perturbation_rate),
            "train_perturbation_noise_std": float(train_perturbation_noise_std),
        }
        if return_summary:
            return summary
        return best_train_auc, best_valid_auc



    def eval(
        self,
        test_data,
        batch_size=None,
        return_summary=False,
        perturbation="clean",
        perturbation_rate=0.0,
        perturbation_noise_std=0.0,
        perturbation_seed=0,
        return_records=False,
    ) -> ...:
        self.dekt_net.eval()
        batch_size = batch_size or self.eval_batch_size
        return test_one_epoch(
            self.dekt_net,
            batch_size,
            *test_data,
            perturbation=perturbation,
            perturbation_rate=perturbation_rate,
            perturbation_noise_std=perturbation_noise_std,
            perturbation_seed=perturbation_seed,
            return_records=return_records,
            return_summary=return_summary,
        )

    def save(self, filepath) -> ...:
        torch.save(self.dekt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath) -> ...:
        self.dekt_net.load_state_dict(torch.load(filepath, map_location=device))
        logging.info("load parameters from %s" % filepath)
