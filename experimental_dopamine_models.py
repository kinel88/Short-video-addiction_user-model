import torch
import torch.nn.functional as F

from train_module import BehaviorFitModel, _softplus_inverse


class PostHistoryPeakRewardDopamineBehaviorFitModel(BehaviorFitModel):
    """
    Experimental dopamine variant for offline-only evaluation.

    Under the new scaffold, it shares the default reward-drive ingredients:
    expected_reward plus session_peak, followed by habit modulation and
    baseline return toward the lower session baseline.
    """

    def make_strict_zero_dopamine_eval_copy(self):
        device = next(self.parameters()).device
        clone = PostHistoryPeakRewardDopamineBehaviorFitModel(
            embedding_dim=self.embedding_dim,
            user_ids=self.user_ids,
        ).to(device=device)
        clone.load_state_dict(self.state_dict())
        clone._strict_zero_dopamine_eval = True
        clone.eval()
        return clone


class IntegratedSignalDopamineBehaviorFitModel(BehaviorFitModel):
    """
    Experimental dopamine variant for offline-only evaluation.

    It keeps the default expected_reward + session_peak reward drive and adds
    current-step score_engagement plus novelty as extra reward ingredients
    before the shared habit modulation and baseline-return scaffold.
    """

    def __init__(self, embedding_dim, user_ids=None):
        super().__init__(embedding_dim=embedding_dim, user_ids=user_ids)
        self.dopamine_novelty_coef_raw = torch.nn.Parameter(
            torch.tensor(_softplus_inverse(0.30), dtype=torch.float32)
        )

    def _params(self):
        params = super()._params()
        params["dopamine_novelty_coef"] = F.softplus(self.dopamine_novelty_coef_raw)
        return params

    def make_strict_zero_dopamine_eval_copy(self):
        device = next(self.parameters()).device
        clone = IntegratedSignalDopamineBehaviorFitModel(
            embedding_dim=self.embedding_dim,
            user_ids=self.user_ids,
        ).to(device=device)
        clone.load_state_dict(self.state_dict())
        clone._strict_zero_dopamine_eval = True
        clone.eval()
        return clone

    def _compute_dopamine_reward_drive(
        self,
        params,
        state,
        new_state,
        aux,
        score_engagement,
    ):
        base_reward_drive = super()._compute_dopamine_reward_drive(
            params=params,
            state=state,
            new_state=new_state,
            aux=aux,
            score_engagement=score_engagement,
        )
        return (
            base_reward_drive
            + params["dopamine_score_engagement_coef"] * score_engagement
            + params["dopamine_novelty_coef"] * aux["novelty_norm"]
        )

    def export_env_params(self):
        params = self._params()
        exported = super().export_env_params()
        exported["dopamine_novelty_coef"] = float(
            params["dopamine_novelty_coef"].detach().cpu().item()
        )
        return exported


class RelaxToHigherBaselineDopamineBehaviorFitModel(BehaviorFitModel):
    """
    Experimental dopamine variant for offline-only evaluation.

    The old moving-higher-baseline rule is retired. Under the new scaffold,
    this variant keeps one extra direct score_engagement ingredient in the
    reward drive while exposing the higher normal dopamine level under the
    legacy `dopamine_true_baseline` name for report compatibility.
    """

    def _params(self):
        params = super()._params()
        params["dopamine_true_baseline"] = params["dopamine_normal_level"]
        return params

    def make_strict_zero_dopamine_eval_copy(self):
        device = next(self.parameters()).device
        clone = RelaxToHigherBaselineDopamineBehaviorFitModel(
            embedding_dim=self.embedding_dim,
            user_ids=self.user_ids,
        ).to(device=device)
        clone.load_state_dict(self.state_dict())
        clone._strict_zero_dopamine_eval = True
        clone.eval()
        return clone

    def _compute_dopamine_reward_drive(
        self,
        params,
        state,
        new_state,
        aux,
        score_engagement,
    ):
        base_reward_drive = super()._compute_dopamine_reward_drive(
            params=params,
            state=state,
            new_state=new_state,
            aux=aux,
            score_engagement=score_engagement,
        )
        return base_reward_drive + score_engagement

    def export_env_params(self):
        params = self._params()
        exported = super().export_env_params()
        exported["dopamine_true_baseline"] = float(
            params["dopamine_true_baseline"].detach().cpu().item()
        )
        return exported
