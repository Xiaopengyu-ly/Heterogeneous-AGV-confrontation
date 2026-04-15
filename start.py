from scripts.train_SAC import train_agent, test_and_vis
from scripts.VQVAE_sample import sampler, data_processer_for_VQVAE, data_processer_for_TwoTower
from scripts.behavior_clone import train_aligned_bc, inject_to_sac
from models.policies.finetune_sac import online_finetune
from models.predictors.agent_dyn_predictor import train_forward_model

if __name__ == "__main__":
    
    online_finetune("./models/policies/sac_policy_spirl")
