# coding: utf-8

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DEKTNet(nn.Module):
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
        n_qd=80,
        n_sd=50,
        n_tp=17,
        n_att=1000,
        dropout=0.2,
        graft_mode="off",
        knowledge_update_mode="auto",
        problem_to_skill=None,
        q_gamma=0.0,
        reliability_mode="off",
        init_rho_bias=3.0,
    ):
        super(DEKTNet, self).__init__()
        self.d_m = d_m
        self.d_k = d_k
        self.d_a = d_a
        self.d_e = d_e
        self.n_question = n_question
        self.graft_mode = graft_mode
        self.knowledge_update_mode = knowledge_update_mode
        self.reliability_mode = reliability_mode

        if graft_mode not in {"off", "affect_blend", "both"}:
            raise ValueError(f"Unsupported graft_mode: {graft_mode}")
        if knowledge_update_mode not in {"auto", "dense", "sparse"}:
            raise ValueError(f"Unsupported knowledge_update_mode: {knowledge_update_mode}")
        if reliability_mode not in {"off", "learned", "fixed_one", "fixed_half"}:
            raise ValueError(f"Unsupported reliability_mode: {reliability_mode}")

        if problem_to_skill is not None:
            self.problem_to_skill = problem_to_skill.long()
            self.q_matrix = None
            self.q_gamma = float(q_gamma)
        else:
            if q_matrix is None:
                raise ValueError("Either q_matrix or problem_to_skill must be provided.")
            self.q_matrix = q_matrix
            self.problem_to_skill = torch.argmax(q_matrix, dim=1).long()
            gamma_candidates = q_matrix[0][q_matrix[0] < 1.0]
            self.q_gamma = float(gamma_candidates[0].item()) if gamma_candidates.numel() > 0 else 0.0


        self.at_embed = nn.Embedding(n_at + 10, d_k)
        torch.nn.init.xavier_uniform_(self.at_embed.weight)
        self.it_embed = nn.Embedding(n_it + 10, d_k)
        torch.nn.init.xavier_uniform_(self.it_embed.weight)
        self.e_embed = nn.Embedding(n_exercise + 10, d_k)
        torch.nn.init.xavier_uniform_(self.e_embed.weight)
        self.s_embed = nn.Embedding(n_question + 10, d_k)
        torch.nn.init.xavier_uniform_(self.s_embed.weight)

        self.att_embed = nn.Embedding(n_att + 10, d_k)
        torch.nn.init.xavier_uniform_(self.att_embed.weight)

        self.fru_embed = nn.Embedding(self.d_m, d_k)
        torch.nn.init.xavier_uniform_(self.fru_embed.weight)
        self.conf_embed = nn.Embedding(self.d_m, d_k)
        torch.nn.init.xavier_uniform_(self.conf_embed.weight)
        self.conc_embed = nn.Embedding(self.d_m, d_k)
        torch.nn.init.xavier_uniform_(self.conc_embed.weight)
        self.bor_embed = nn.Embedding(self.d_m, d_k)
        torch.nn.init.xavier_uniform_(self.bor_embed.weight)


        self.sd_embed = nn.Embedding(n_sd + 10, d_k)
        torch.nn.init.xavier_uniform_(self.sd_embed.weight)
        self.qd_embed = nn.Embedding(n_qd + 10, d_k)
        torch.nn.init.xavier_uniform_(self.qd_embed.weight)
        self.tp_embed = nn.Embedding(n_tp + 10, d_k)
        torch.nn.init.xavier_uniform_(self.tp_embed.weight)


        self.linear_1 = nn.Linear(4*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_1.weight) 
        self.linear_2 = nn.Linear(2*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        self.linear_3 = nn.Linear(3*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_3.weight)
        self.linear_4 = nn.Linear(3 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_4.weight)
        self.linear_5 = nn.Linear(6*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_5.weight)
        self.response_head = nn.Linear(d_k, 1)
        torch.nn.init.xavier_uniform_(self.response_head.weight)

        self.linear_a = nn.Linear(4*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_a.weight)
        self.linear_e = nn.Linear(2*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_e.weight)

        self.linear_attblock = nn.Linear(d_k*5, d_k)
        torch.nn.init.xavier_uniform_(self.linear_attblock.weight)


        self.linear_emo = nn.Linear(4*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_emo.weight)

        self.linear_affect_blend = nn.Linear(2 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_affect_blend.weight)
        self.affect_blend_logit = nn.Parameter(torch.tensor(2.0))


        if d_k % 4 != 0:
            raise ValueError("d_k must be divisible by 4 for the four affect prediction heads.")
        emotion_head_dim = d_k // 4
        self.linear_bor = nn.Linear(emotion_head_dim, 1)
        torch.nn.init.xavier_uniform_(self.linear_bor.weight)
        self.linear_conc = nn.Linear(emotion_head_dim, 1)
        torch.nn.init.xavier_uniform_(self.linear_conc.weight)
        self.linear_conf = nn.Linear(emotion_head_dim, 1)
        torch.nn.init.xavier_uniform_(self.linear_conf.weight)
        self.linear_fru = nn.Linear(emotion_head_dim, 1)
        torch.nn.init.xavier_uniform_(self.linear_fru.weight)


        self.linear_siga = nn.Linear(4*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_siga.weight)
        self.linear_tana = nn.Linear(4*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_tana.weight)

        self.response_residual_1 = nn.Linear(4 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.response_residual_1.weight)
        self.response_residual_2 = nn.Linear(d_k, d_k)
        torch.nn.init.xavier_uniform_(self.response_residual_2.weight)
        self.response_residual_scale = nn.Parameter(torch.tensor(0.0))

        reliability_input_dim = 13 + 2 * d_k
        self.reliability_hidden = nn.Linear(reliability_input_dim, d_k)
        torch.nn.init.xavier_uniform_(self.reliability_hidden.weight)
        self.reliability_out = nn.Linear(d_k, 1)
        torch.nn.init.zeros_(self.reliability_out.weight)
        torch.nn.init.constant_(self.reliability_out.bias, init_rho_bias)

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.rulu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        


    def _map_to_class(self, tensor):
        safe_tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        scaled = torch.floor(safe_tensor * self.d_m).long()
        return torch.clamp(scaled, min=0, max=self.d_m - 1)

    def _affect_values(self, bor_data, conc_data, conf_data, fru_data):
        affect_values = torch.stack(
            (
                bor_data.float(),
                conc_data.float(),
                conf_data.float(),
                fru_data.float(),
            ),
            dim=2,
        )
        return torch.nan_to_num(affect_values, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def _causal_affect_baseline(self, affect_values, valid_mask):
        valid = valid_mask.float().unsqueeze(2)
        cumulative_count = valid.cumsum(dim=1).clamp(min=1.0)
        cumulative_sum = (affect_values * valid).cumsum(dim=1)
        return cumulative_sum / cumulative_count

    def _affect_embedding_from_values(self, affect_values):
        if self.reliability_mode in {"learned", "fixed_half"}:
            bor = self._soft_affect_channel_embedding(affect_values[:, 0], self.bor_embed)
            conc = self._soft_affect_channel_embedding(affect_values[:, 1], self.conc_embed)
            conf = self._soft_affect_channel_embedding(affect_values[:, 2], self.conf_embed)
            fru = self._soft_affect_channel_embedding(affect_values[:, 3], self.fru_embed)
        else:
            bor = self.bor_embed(self._map_to_class(affect_values[:, 0]))
            conc = self.conc_embed(self._map_to_class(affect_values[:, 1]))
            conf = self.conf_embed(self._map_to_class(affect_values[:, 2]))
            fru = self.fru_embed(self._map_to_class(affect_values[:, 3]))
        return self.linear_emo(torch.cat((fru, conf, conc, bor), 1))

    def _soft_affect_channel_embedding(self, values, embedding):
        safe_values = torch.nan_to_num(values.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        scaled = safe_values * (self.d_m - 1)
        lower = torch.floor(scaled).long().clamp(min=0, max=self.d_m - 1)
        upper = (lower + 1).clamp(max=self.d_m - 1)
        fraction = (scaled - lower.float()).unsqueeze(1)
        return (1.0 - fraction) * embedding(lower) + fraction * embedding(upper)

    def _trusted_affect(self, current_affect, baseline_affect, h_tilde_pre, affect_h_pre, answer):
        deviation = torch.abs(current_affect - baseline_affect)
        if self.reliability_mode in {"off", "fixed_one"}:
            rho = torch.ones(current_affect.size(0), 1, device=current_affect.device)
        elif self.reliability_mode == "fixed_half":
            rho = torch.full((current_affect.size(0), 1), 0.5, device=current_affect.device)
        else:
            features = torch.cat(
                (
                    current_affect,
                    baseline_affect,
                    deviation,
                    answer.float().view(-1, 1),
                    h_tilde_pre,
                    affect_h_pre,
                ),
                1,
            )
            rho = self.sig(self.reliability_out(self.dropout(self.rulu(self.reliability_hidden(features)))))
        trusted = rho * current_affect + (1.0 - rho) * baseline_affect
        return trusted.clamp(0.0, 1.0), rho, deviation


    def _blend_affect_embedding(self, emo, affect_state):
        if self.graft_mode not in {"affect_blend", "both"}:
            return emo

        keep_weight = torch.sigmoid(self.affect_blend_logit)
        blended = self.tanh(self.linear_affect_blend(torch.cat((emo, affect_state), 1)))
        return keep_weight * emo + (1.0 - keep_weight) * blended


    def _apply_response_residual(self, base_logits, update_h, affect_h, block_b, x):
        if self.graft_mode != "both":
            return base_logits

        residual_hidden = self.tanh(
            self.response_residual_1(torch.cat((update_h, affect_h, block_b, x), 1))
        )
        residual_hidden = self.dropout(residual_hidden)
        residual_logits = self.response_residual_2(residual_hidden)
        residual_scale = torch.tanh(self.response_residual_scale)
        return base_logits + residual_scale * residual_logits


    def _lookup_skill_indices(self, exercise_ids):
        return self.problem_to_skill[exercise_ids]


    def _read_knowledge_state(self, knowledge_state, exercise_ids):
        batch_indices = torch.arange(exercise_ids.size(0), device=exercise_ids.device)
        skill_indices = self._lookup_skill_indices(exercise_ids)
        selected_state = knowledge_state[batch_indices, skill_indices]
        if self.q_gamma == 0.0:
            return selected_state
        return self.q_gamma * knowledge_state.sum(dim=1) + (1.0 - self.q_gamma) * selected_state


    def _expand_learning_gain(self, learning_gain, exercise_ids, n_skill):
        if self.q_gamma == 0.0:
            lg_tilde = torch.zeros(
                learning_gain.size(0),
                n_skill,
                self.d_k,
                device=learning_gain.device,
            )
            selected_scale = 1.0
        else:
            lg_tilde = torch.full(
                (learning_gain.size(0), n_skill, self.d_k),
                self.q_gamma,
                device=learning_gain.device,
            ) * learning_gain.unsqueeze(1)
            selected_scale = 1.0 - self.q_gamma

        skill_indices = self._lookup_skill_indices(exercise_ids).view(-1, 1, 1).expand(-1, 1, self.d_k)
        lg_tilde.scatter_add_(1, skill_indices, selected_scale * learning_gain.unsqueeze(1))
        return self.dropout(lg_tilde)


    def _use_sparse_update(self):
        if self.knowledge_update_mode == "sparse":
            return True
        if self.knowledge_update_mode == "dense":
            return False
        return self.n_question > 5000


    def _sparse_knowledge_update(self, h_pre, learning_gain, interval_repr, exercise_ids):
        batch_indices = torch.arange(exercise_ids.size(0), device=exercise_ids.device)
        skill_indices = self._lookup_skill_indices(exercise_ids)
        selected_state = h_pre[batch_indices, skill_indices]
        gamma_f_selected = self.sig(self.linear_4(torch.cat((selected_state, learning_gain, interval_repr), 1)))
        updated_selected = learning_gain + gamma_f_selected * selected_state
        h = h_pre.clone()
        h[batch_indices, skill_indices] = updated_selected
        return h


    def forward(
        self,
        e_data,
        s_data,
        at_data,
        a_data,
        it_data,
        bor_data,
        conc_data,
        conf_data,
        fru_data,
        qd_data,
        sd_data,
        tp_data,
        att_data,
        return_details=False,
    ):
        batch_size, seq_len = e_data.size(0), e_data.size(1)

        e_embed_data = self.e_embed(e_data)
        s_embed_data = self.s_embed(s_data)
        at_embed_data = self.at_embed(at_data)
        it_embed_data = self.it_embed(it_data)

        qd_embed_data = self.qd_embed(qd_data)
        sd_embed_data = self.sd_embed(sd_data)
        tp_embed_data = self.tp_embed(tp_data)
        att_embed_data = self.att_embed(att_data)

        a_embedd_data = a_data.view(-1, 1).repeat(1, self.d_k).view(batch_size, -1, self.d_k)
        affect_values = self._affect_values(bor_data, conc_data, conf_data, fru_data)
        affect_baseline = self._causal_affect_baseline(affect_values, e_data > 0)
        attblock = self.linear_attblock(torch.cat((e_embed_data,s_embed_data,qd_embed_data,sd_embed_data,tp_embed_data), 2))
        all_learning = self.linear_1(torch.cat((at_embed_data,s_embed_data,a_embedd_data,e_embed_data), 2))
        

        # Use deterministic zero initial states instead of sampling a new random state on every forward pass.
        h_pre = torch.zeros(batch_size, self.n_question, self.d_k, device=e_data.device)
        h_tilde_pre = None
        affect_h_pre = torch.zeros(batch_size, self.d_k, device=e_data.device)


        pred      = torch.zeros(batch_size, seq_len, device=e_data.device)
        pred_bor  = torch.zeros(batch_size, seq_len, device=e_data.device)
        pred_conc = torch.zeros(batch_size, seq_len, device=e_data.device)
        pred_conf = torch.zeros(batch_size, seq_len, device=e_data.device)
        pred_fru  = torch.zeros(batch_size, seq_len, device=e_data.device)
        rho_data = torch.ones(batch_size, seq_len, device=e_data.device)
        trusted_affect_data = torch.zeros(batch_size, seq_len, 4, device=e_data.device)
        affect_deviation_data = torch.zeros(batch_size, seq_len, 4, device=e_data.device)


        for t in range(0, seq_len - 1):
            e = e_data[:, t]
            if h_tilde_pre is None:
                h_tilde_pre = self._read_knowledge_state(h_pre, e)

            a = a_embedd_data[:,t]
            it = it_embed_data[:, t]
            at = at_embed_data[:,t]
            block_a = attblock[:,t] 
            block_b = attblock[:,t+1]        
            trusted_affect, rho, affect_deviation = self._trusted_affect(
                affect_values[:, t],
                affect_baseline[:, t],
                h_tilde_pre,
                affect_h_pre,
                a_data[:, t],
            )
            rho_data[:, t] = rho.squeeze(1)
            trusted_affect_data[:, t] = trusted_affect
            affect_deviation_data[:, t] = affect_deviation
            emo = self._blend_affect_embedding(self._affect_embedding_from_values(trusted_affect), affect_h_pre)
            a_e_d = att_embed_data[:,t]

            # es_t block
            relation_matirx = torch.stack([a,a_e_d,at,it],axis=1)
            correlation_matrix = torch.sum(emo.unsqueeze(1) * relation_matirx, dim=2)
            softmax_result = F.softmax(correlation_matrix, dim=1)
            result = torch.matmul(softmax_result.unsqueeze(1), relation_matirx).squeeze(dim=1)


            # 1. Knowledge State Boosting Module 
            learning = all_learning[:, t]
            learning_gain = self.linear_2(torch.cat((learning, h_tilde_pre), 1))
            learning_gain = self.tanh(learning_gain)
            gamma_l = self.linear_3(torch.cat(( learning,h_tilde_pre,emo), 1))
            gamma_l = self.sig(gamma_l)
            LG = gamma_l * ((learning_gain + 1) / 2)
            if self._use_sparse_update():
                h = self._sparse_knowledge_update(h_pre, LG, it, e)
            else:
                LG_tilde = self._expand_learning_gain(LG, e, h_pre.size(1))
                n_skill1 = LG_tilde.size(1)
                gamma_f = self.sig(self.linear_4(torch.cat((
                    h_pre,
                    LG.repeat(1, n_skill1).view(batch_size, -1, self.d_k),
                    it.repeat(1, n_skill1).view(batch_size, -1, self.d_k)
                ), 2)))
                h = LG_tilde + gamma_f * h_pre


            # 2. Emotional State Tracing Module

            affect = self.sig(self.linear_a(torch.cat((emo,block_a,result,a), 1)))
            fa = self.linear_tana(torch.cat((affect,result,a,affect_h_pre), 1))
            fa_gain = self.tanh(fa)
            gt = self.linear_siga(torch.cat(( affect,result,a,affect_h_pre), 1))
            gt_l = self.sig(gt)
            FLG = gt_l * fa_gain
            w1 = F.softmax(LG*FLG, dim=1)
            affect_h = (1-w1)*FLG + w1*affect_h_pre


            # 3. Emotion Prediction Based on Personalized Emotional State
            x = self.sig(self.linear_e(torch.cat(( affect_h , block_b ), 1)))

            x_four = x.view(batch_size,4,-1)
            bor_x  = torch.squeeze( x_four[:,0,:], dim=1)
            conc_x = torch.squeeze( x_four[:,1,:], dim=1)
            conf_x = torch.squeeze( x_four[:,2,:], dim=1)
            fru_x  = torch.squeeze( x_four[:,3,:], dim=1)

           
            x_bor = self.sig(self.linear_bor(bor_x)).squeeze(1)
            x_conc= self.sig(self.linear_conc(conc_x)).squeeze(1)
            x_conf= self.sig(self.linear_conf(conf_x)).squeeze(1)
            x_fru = self.sig(self.linear_fru(fru_x)).squeeze(1)


            fru = self.fru_embed(self._map_to_class(x_fru))
            conf = self.conf_embed(self._map_to_class(x_conf))
            conc = self.conc_embed(self._map_to_class(x_conc))
            bor = self.bor_embed(self._map_to_class(x_bor))


            # 4. Emotion-Boosted Response Prediction
            h_tilde = self._read_knowledge_state(h, e_data[:, t + 1])
            condition = (x > 0.90) | (x < 0.1)  # 创建一个布尔张量，标记极端情绪
            update_h = torch.where(condition,  h_tilde, h_tilde*x)  
            base_logits = self.linear_5(torch.cat((fru,conf ,conc,bor , update_h,block_b*x), 1))
            final_logits = self._apply_response_residual(base_logits, update_h, affect_h, block_b, x)
            y = self.sig(self.response_head(final_logits)).squeeze(1)

            pred[:, t + 1]   =  y
            pred_bor[:,t+1]  = x_bor 
            pred_conc[:,t+1] = x_conc
            pred_conf[:,t+1] = x_conf
            pred_fru[:,t+1]  = x_fru

            # prepare for next prediction
            h_pre = h
            h_tilde_pre = h_tilde
            affect_h_pre = affect_h

        if return_details:
            return pred, pred_bor, pred_conc, pred_conf, pred_fru, {
                "rho": rho_data,
                "trusted_affect": trusted_affect_data,
                "affect_deviation": affect_deviation_data,
                "affect_baseline": affect_baseline,
            }
        return pred , pred_bor, pred_conc, pred_conf, pred_fru
