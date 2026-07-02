from pathlib import Path
import wandb  # <-- WandB importeren
from wandb.integration.sb3 import WandbCallback  # <-- Stable-Baselines3 koppeling importeren
from learning_machines.rl_robobo_env import RoboboObstacleAvoidanceEnv, RoboboObstacleEnvConfig
from stable_baselines3 import DDPG  
from stable_baselines3.common.monitor import Monitor

def train_simple():
    # 1. Initialiseer WandB met jullie workspace (entity) en projectnaam
    wandb.init(
        project="learning-machines",
        entity="Learningmachine",
        mode="offline",  # <-- TIJDELIJK OFFLINE VOOR DE PERFORMANCE TEST
        sync_tensorboard=True,
        name="DDPG-Simple-Run",
    )

    # We gebruiken exact dezelfde camera-loze config als nu
    config = RoboboObstacleEnvConfig(
        max_wheel_speed=100,
        step_millis=20,
        max_episode_steps=500,
        max_ir_value=400.0,
        obstacle_penalty_threshold=0.15,
        collision_ir_threshold=0.85,
        reset_settle_seconds=0.1,
    )

    # Omgeving opstarten
    env = Monitor(RoboboObstacleAvoidanceEnv(config=config))

    # DDPG model definitie (met de MultiInputPolicy fix!)
    model = DDPG(
        "MultiInputPolicy",
        env,
        learning_rate=1e-3,
        batch_size=64,
        tensorboard_log="./tensorboard_logs/",
        verbose=1
    )

    print("Snel trainen met een simpeler model (100.000 stappen)...")
    
    # 2. Geef de WandbCallback mee zodat alle stats live naar jullie dashboard vliegen
    model.learn(
            total_timesteps=100_000, 
            callback=WandbCallback(), 
            progress_bar=True 
        )

    # Model opslaan
    model.save("models/robobo_simple_ddpg")
    print("Klaar! Model opgeslagen als robobo_simple_ddpg.zip")
    
    # 3. Sluit de WandB run netjes af
    wandb.finish()


def test_simple():
    print("--- START TEST RUN ---")
    
    config = RoboboObstacleEnvConfig(
        max_wheel_speed=100,
        step_millis=200,
        max_episode_steps=500,
        max_ir_value=400.0,
        obstacle_penalty_threshold=0.15,
        collision_ir_threshold=0.85,
        reset_settle_seconds=0.1,
    )
    env = Monitor(RoboboObstacleAvoidanceEnv(config=config))

    # Laad jouw opgeslagen DDPG model in
    model = DDPG.load("models/robobo_simple_ddpg")

    # Laat de robot 3 episodes testrijden
    for episode in range(3):
        obs, _ = env.reset()
        done = False
        score = 0
        steps = 0
        
        print(f"\nStart Episode {episode + 1}")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            steps += 1
            
            if steps % 10 == 0:
                print(f"Stap: {steps} | Totale Reward tot nu toe: {score:.2f}")
                
        print(f"Episode {episode + 1} afgelopen na {steps} stappen. Totale score: {score:.2f}")

if __name__ == "__main__":
    train_simple()
