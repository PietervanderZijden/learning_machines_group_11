import gymnasium as gym
import numpy as np

class RoboboGitstormWrapper(gym.Wrapper):
    """
    Deze wrapper doet drie dingen voor de Robobo in GIT-STORM:
    1. Vertaalt 25 discrete knoppen naar continue wielsnelheden [-1.0, 1.0].
    2. Tekent de 8 IR-sensoren als een hotbar op de bovenste 4 pixels van het beeld.
    3. Levert uitsluitend het beeld (H, W, C) terug, wat de GIT-STORM transformer verwacht.
    """
    def __init__(self, env, hotbar_height=4):
        super().__init__(env)
        self.hotbar_height = hotbar_height
        
        # 1. 25 discrete acties (5 standen per wiel)
        self.action_space = gym.spaces.Discrete(25)
        self.bins = [-1.0, -0.5, 0.0, 0.5, 1.0]

        self.frame_num = 0

        # Override observation space naar pure beelden: (64, 64, 3)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )

    def decode_action(self, action):
        """Vertaal een integer (0-24) naar [linker_wiel, rechter_wiel]"""
        # AsyncVectorEnv levert soms een array terug, we pakken de int
        if isinstance(action, (np.ndarray, list)):
            action = int(action[0])
            
        left_idx = action // 5
        right_idx = action % 5
        return np.array([self.bins[left_idx], self.bins[right_idx]], dtype=np.float32)

    def _process_obs(self, obs_dict):
        """Voeg de hotbar toe en transponeer naar (H, W, C)"""
        # Verwacht origineel beeld van (3, 64, 64)
        img = obs_dict["image"].copy()
        ir = obs_dict["ir"]

        # IR naar 0-255 schalen
        ir_normalized = (np.clip(ir, 0.0, 1.0) * 255).astype(np.uint8)

        channels, height, width = img.shape
        pixels_per_sensor = width // 8

        # Teken de hotbar
        for i in range(8):
            start_w = i * pixels_per_sensor
            end_w = start_w + pixels_per_sensor
            # Teken de IR-waarde op alle 3 de kanalen (R, G, B) voor een grijswaarde hotbar
            img[0, :self.hotbar_height, start_w:end_w] = ir_normalized[i]
            img[1, :self.hotbar_height, start_w:end_w] = ir_normalized[i]
            img[2, :self.hotbar_height, start_w:end_w] = ir_normalized[i]

        # GIT-STORM verwacht beelden als (Hoogte, Breedte, Kanalen)
        img_hwc = np.transpose(img, (1, 2, 0))
        return img_hwc

    def reset(self, **kwargs):
        self.frame_num = 0
        obs_dict, info = self.env.reset(**kwargs)
        obs = self._process_obs(obs_dict)
        
        # Voeg verplichte GIT-STORM info keys toe
        info["life_loss"] = False
        info["episode_frame_number"] = np.array([self.frame_num])
        
        return obs, info

    def step(self, action):
        cont_action = self.decode_action(action)
        obs_dict, reward, terminated, truncated, info = self.env.step(cont_action)

        obs = self._process_obs(obs_dict)
        self.frame_num += 1

        # Voeg verplichte GIT-STORM info keys toe
        info["life_loss"] = False
        info["episode_frame_number"] = np.array([self.frame_num])

        return obs, reward, terminated, truncated, info


def build_robobo_vec_env(env_config, num_envs=1, ranges=None):
    """
    Vervangt de originele build_vec_env uit GIT-STORM.
    """
    from learning_machines.rl_robobo_compact_env import RoboboCompactEnv
    from learning_machines.domain_randomization import DomainRandomizationWrapper

    def make_env():
        # 1. Start de basis robot omgeving
        rob_env = RoboboCompactEnv(config=env_config)
        
        # 2. Domain randomization (voor kleuren en belichting)
        rand_env = DomainRandomizationWrapper(
            rob_env,
            enabled=(ranges is not None),
            ranges=ranges,
        )
        
        # 3. Onze GIT-STORM Hotbar + Knoppen wrapper
        gitstorm_env = RoboboGitstormWrapper(rand_env)
        return gitstorm_env

    # We draaien de omgeving in een AsyncVectorEnv, wat GIT-STORM standaard vereist
    env_fns = [make_env for _ in range(num_envs)]
    vec_env = gym.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env